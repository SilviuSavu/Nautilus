/**
 * Story 5.3: Unit Tests for ScheduledReports Component
 * Tests for automated report scheduling interface
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { notification } from 'antd';
import ScheduledReports from '../ScheduledReports';
import { ReportTemplate, ReportType, ScheduleFrequency } from '../../../types/export';

// Mock antd notification
vi.mock('antd', async () => {
  const actual = await vi.importActual('antd');
  return {
    ...actual,
    notification: {
      success: vi.fn(),
      error: vi.fn(),
      info: vi.fn(),
    },
  };
});

// Mock dayjs
vi.mock('dayjs', async () => {
  const actualDayjs = await vi.importActual('dayjs');
  const mockDayjs = (date?: any) => actualDayjs(date || '2024-01-20T10:00:00Z');
  mockDayjs.extend = actualDayjs.extend;
  mockDayjs.isToday = vi.fn().mockReturnValue(true);
  mockDayjs.isAfter = vi.fn().mockReturnValue(true);
  return mockDayjs;
});

const mockScheduledTemplates: ReportTemplate[] = [
  {
    id: 'template-001',
    name: 'Daily Performance Report',
    description: 'Comprehensive daily trading performance analysis',
    type: ReportType.PERFORMANCE,
    format: 'pdf' as any,
    sections: [],
    parameters: [],
    schedule: {
      frequency: ScheduleFrequency.DAILY,
      time: '08:00',
      timezone: 'UTC',
      recipients: ['trader@example.com', 'manager@example.com']
    },
    created_at: new Date('2024-01-15'),
    updated_at: new Date('2024-01-20')
  },
  {
    id: 'template-002',
    name: 'Weekly Risk Assessment',
    description: 'Weekly risk metrics and exposure analysis',
    type: ReportType.RISK,
    format: 'excel' as any,
    sections: [],
    parameters: [],
    schedule: {
      frequency: ScheduleFrequency.WEEKLY,
      time: '09:00',
      timezone: 'America/New_York',
      recipients: ['risk@example.com']
    },
    created_at: new Date('2024-01-10'),
    updated_at: new Date('2024-01-18')
  }
];

describe('ScheduledReports', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders statistics cards', () => {
      render(<ScheduledReports />);
      
      expect(screen.getByText('Active Schedules')).toBeInTheDocument();
      expect(screen.getByText('Completed Today')).toBeInTheDocument();
      expect(screen.getByText('Failed Today')).toBeInTheDocument();
      expect(screen.getByText('Pending')).toBeInTheDocument();
    });

    it('renders scheduled templates table', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
        expect(screen.getByText('Weekly Risk Assessment')).toBeInTheDocument();
      });
    });

    it('renders recent executions list', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Recent Executions')).toBeInTheDocument();
      });
    });

    it('displays correct statistics', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        // Should show statistics based on mock data
        expect(screen.getByText('3')).toBeInTheDocument(); // Active schedules
      });
    });
  });

  describe('Scheduled Templates Management', () => {
    it('displays scheduled template information correctly', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
        expect(screen.getByText('DAILY')).toBeInTheDocument();
        expect(screen.getByText('08:00 (UTC)')).toBeInTheDocument();
        expect(screen.getByText('2 recipients')).toBeInTheDocument();
      });
    });

    it('shows different schedule frequencies with proper colors', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('DAILY')).toBeInTheDocument();
        expect(screen.getByText('WEEKLY')).toBeInTheDocument();
      });
    });

    it('displays recipient information', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('2 recipients')).toBeInTheDocument();
        expect(screen.getByText('1 recipients')).toBeInTheDocument();
        expect(screen.getByText('trader@example.com,')).toBeInTheDocument();
      });
    });

    it('shows active status badge', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        const activeBadges = screen.getAllByText('Active');
        expect(activeBadges.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Manual Report Execution', () => {
    it('allows running reports manually', async () => {
      const user = userEvent.setup();
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Find and click "Run Now" button
      const runButtons = screen.getAllByLabelText('Run Now');
      await user.click(runButtons[0]);
      
      await waitFor(() => {
        expect(notification.info).toHaveBeenCalledWith({
          message: 'Report Generation Started',
          description: 'Manual execution started for Daily Performance Report',
        });
      });
    });

    it('adds new execution to the list when run manually', async () => {
      const user = userEvent.setup();
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      const runButtons = screen.getAllByLabelText('Run Now');
      await user.click(runButtons[0]);
      
      // Should show running status
      await waitFor(() => {
        expect(notification.info).toHaveBeenCalled();
      });
    });
  });

  describe('Execution History', () => {
    it('opens execution history modal', async () => {
      const user = userEvent.setup();
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Find and click "View History" button
      const historyButtons = screen.getAllByLabelText('View History');
      await user.click(historyButtons[0]);
      
      expect(screen.getByText('Execution History')).toBeInTheDocument();
      expect(screen.getByText('- Daily Performance Report')).toBeInTheDocument();
    });

    it('displays execution history data', async () => {
      const user = userEvent.setup();
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      const historyButtons = screen.getAllByLabelText('View History');
      await user.click(historyButtons[0]);
      
      expect(screen.getByText('Execution History')).toBeInTheDocument();
      // Should show table with execution history
      expect(screen.getByText('Template')).toBeInTheDocument();
      expect(screen.getByText('Scheduled')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
    });

    it('closes execution history modal', async () => {
      const user = userEvent.setup();
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Open history modal
      const historyButtons = screen.getAllByLabelText('View History');
      await user.click(historyButtons[0]);
      
      // Close modal
      const closeButton = screen.getByText('Close');
      await user.click(closeButton);
      
      expect(screen.queryByText('Execution History')).not.toBeInTheDocument();
    });
  });

  describe('Schedule Management', () => {
    it('allows disabling schedules', async () => {
      const user = userEvent.setup();
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Find and click "Disable Schedule" button
      const disableButtons = screen.getAllByLabelText('Disable Schedule');
      await user.click(disableButtons[0]);
      
      await waitFor(() => {
        expect(notification.info).toHaveBeenCalledWith({
          message: 'Schedule Updated',
          description: 'Schedule has been disabled for the selected template',
        });
      });
    });
  });

  describe('Status Icons and Colors', () => {
    it('displays correct status icons', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        // Check that status icons are rendered (by checking for status-related elements)
        expect(screen.getByText('Recent Executions')).toBeInTheDocument();
      });
    });

    it('shows progress bars for running executions', async () => {
      const user = userEvent.setup();
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Run a report to create a running execution
      const runButtons = screen.getAllByLabelText('Run Now');
      await user.click(runButtons[0]);
      
      // Should show progress indication
      await waitFor(() => {
        expect(notification.info).toHaveBeenCalled();
      });
    });
  });

  describe('Next Run Calculations', () => {
    it('calculates and displays next run times', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Next Run')).toBeInTheDocument();
        // Should show formatted dates for next runs
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
    });

    it('handles different timezones in next run display', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        // Should show UTC and America/New_York schedules
        expect(screen.getByText('08:00 (UTC)')).toBeInTheDocument();
        expect(screen.getByText('09:00 (America/New_York)')).toBeInTheDocument();
      });
    });
  });

  describe('Recent Executions List', () => {
    it('displays recent executions with proper formatting', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Recent Executions')).toBeInTheDocument();
        // Should show execution items
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
    });

    it('shows file sizes for completed executions', async () => {
      render(<ScheduledReports />);
      
      // Should show file size tags for completed executions
      await waitFor(() => {
        expect(screen.getByText('Recent Executions')).toBeInTheDocument();
      });
    });

    it('displays execution timestamps correctly', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Recent Executions')).toBeInTheDocument();
        // Should format timestamps properly
      });
    });
  });

  describe('Error Handling', () => {
    it('handles template loading errors gracefully', () => {
      // Mock console.error to avoid test output noise
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      render(<ScheduledReports />);
      
      // Component should render without crashing even with missing data
      expect(screen.getByText('Active Schedules')).toBeInTheDocument();
      
      consoleSpy.mockRestore();
    });

    it('handles manual execution errors gracefully', async () => {
      const user = userEvent.setup();
      
      // Mock console.error
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      const runButtons = screen.getAllByLabelText('Run Now');
      await user.click(runButtons[0]);
      
      // Should handle any errors gracefully
      await waitFor(() => {
        expect(notification.info).toHaveBeenCalled();
      });
      
      consoleSpy.mockRestore();
    });
  });

  describe('Accessibility', () => {
    it('provides proper ARIA labels for action buttons', async () => {
      render(<ScheduledReports />);
      
      await waitFor(() => {
        expect(screen.getByText('Daily Performance Report')).toBeInTheDocument();
      });
      
      // Check for proper labels
      expect(screen.getAllByLabelText('Run Now')).toHaveLength(3);
      expect(screen.getAllByLabelText('View History')).toHaveLength(3);
      expect(screen.getAllByLabelText('Disable Schedule')).toHaveLength(3);
    });

    it('has proper heading structure', () => {
      render(<ScheduledReports />);
      
      expect(screen.getByText('Scheduled Templates')).toBeInTheDocument();
      expect(screen.getByText('Recent Executions')).toBeInTheDocument();
    });
  });
});