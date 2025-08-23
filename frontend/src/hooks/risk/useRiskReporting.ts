import { useState, useEffect, useCallback, useRef } from 'react';
import { riskService } from '../../components/Risk/services/riskService';

export interface RiskReport {
  id: string;
  portfolio_id: string;
  report_type: 'daily' | 'weekly' | 'monthly' | 'custom' | 'compliance';
  format: 'json' | 'pdf' | 'excel' | 'html';
  generated_at: Date;
  period_start: Date;
  period_end: Date;
  status: 'generating' | 'completed' | 'failed' | 'scheduled';
  file_url?: string;
  file_size_bytes?: number;
  sections_included: ReportSection[];
  metadata: {
    total_pages?: number;
    chart_count?: number;
    table_count?: number;
    generation_time_seconds?: number;
  };
  error_message?: string;
}

export interface ReportSection {
  id: string;
  name: string;
  type: 'summary' | 'metrics' | 'chart' | 'table' | 'analysis' | 'compliance';
  enabled: boolean;
  configuration?: Record<string, any>;
}

export interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  report_type: RiskReport['report_type'];
  sections: ReportSection[];
  default_format: RiskReport['format'];
  frequency?: 'daily' | 'weekly' | 'monthly';
  auto_schedule?: boolean;
  recipients?: string[];
  created_by: string;
  created_at: Date;
  is_system_template: boolean;
}

export interface ReportSchedule {
  id: string;
  template_id: string;
  portfolio_id: string;
  frequency: 'daily' | 'weekly' | 'monthly';
  time_of_day: string; // HH:MM format
  day_of_week?: number; // 0-6, Sunday = 0
  day_of_month?: number; // 1-31
  enabled: boolean;
  next_run: Date;
  last_run?: Date;
  recipients: string[];
  delivery_method: 'email' | 'webhook' | 'storage';
}

export interface RiskReportingState {
  reports: RiskReport[];
  templates: ReportTemplate[];
  schedules: ReportSchedule[];
  activeGenerations: Set<string>;
  loading: {
    reports: boolean;
    templates: boolean;
    schedules: boolean;
    generating: boolean;
  };
  error: string | null;
  lastUpdated: Date | null;
}

export interface UseRiskReportingProps {
  portfolioId: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export interface GenerateReportParams {
  report_type: RiskReport['report_type'];
  format: RiskReport['format'];
  period_start: Date;
  period_end: Date;
  sections?: ReportSection[];
  template_id?: string;
  include_stress_tests?: boolean;
  include_scenarios?: boolean;
  custom_parameters?: Record<string, any>;
}

export const useRiskReporting = ({
  portfolioId,
  autoRefresh = true,
  refreshInterval = 30000 // 30 seconds
}: UseRiskReportingProps) => {
  const [state, setState] = useState<RiskReportingState>({
    reports: [],
    templates: [],
    schedules: [],
    activeGenerations: new Set(),
    loading: {
      reports: true,
      templates: true,
      schedules: true,
      generating: false
    },
    error: null,
    lastUpdated: null
  });

  const refreshTimeoutRef = useRef<NodeJS.Timeout>();
  const pollTimeoutRef = useRef<NodeJS.Timeout>();

  const updateState = useCallback((updates: Partial<RiskReportingState>) => {
    setState(prev => ({
      ...prev,
      ...updates,
      loading: { ...prev.loading, ...updates.loading }
    }));
  }, []);

  const fetchReports = useCallback(async (silent = false) => {
    if (!silent) {
      updateState({ loading: { ...state.loading, reports: true } });
    }

    try {
      // Mock API call - would be replaced with actual Sprint 3 backend endpoint
      const mockReports: RiskReport[] = [
        {
          id: 'report-1',
          portfolio_id: portfolioId,
          report_type: 'daily',
          format: 'pdf',
          generated_at: new Date(Date.now() - 86400000), // Yesterday
          period_start: new Date(Date.now() - 172800000), // 2 days ago
          period_end: new Date(Date.now() - 86400000), // Yesterday
          status: 'completed',
          file_url: '/api/v1/risk/reports/report-1/download',
          file_size_bytes: 2543210,
          sections_included: [
            { id: 'summary', name: 'Executive Summary', type: 'summary', enabled: true },
            { id: 'var-metrics', name: 'VaR Metrics', type: 'metrics', enabled: true },
            { id: 'exposure-chart', name: 'Exposure Analysis', type: 'chart', enabled: true }
          ],
          metadata: {
            total_pages: 12,
            chart_count: 8,
            table_count: 5,
            generation_time_seconds: 23.4
          }
        }
      ];

      updateState({
        reports: mockReports,
        error: null,
        lastUpdated: new Date(),
        loading: { ...state.loading, reports: false }
      });

      return mockReports;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, reports: false }
        });
      }
      throw error;
    }
  }, [portfolioId, updateState, state.loading]);

  const fetchTemplates = useCallback(async (silent = false) => {
    if (!silent) {
      updateState({ loading: { ...state.loading, templates: true } });
    }

    try {
      // Mock API call - would be replaced with actual Sprint 3 backend endpoint
      const mockTemplates: ReportTemplate[] = [
        {
          id: 'template-daily-standard',
          name: 'Standard Daily Risk Report',
          description: 'Comprehensive daily risk overview with key metrics and exposures',
          report_type: 'daily',
          sections: [
            { id: 'summary', name: 'Executive Summary', type: 'summary', enabled: true },
            { id: 'var-metrics', name: 'VaR Metrics', type: 'metrics', enabled: true },
            { id: 'exposure-breakdown', name: 'Exposure Breakdown', type: 'table', enabled: true },
            { id: 'correlation-matrix', name: 'Correlation Matrix', type: 'chart', enabled: true },
            { id: 'limit-status', name: 'Risk Limit Status', type: 'table', enabled: true }
          ],
          default_format: 'pdf',
          frequency: 'daily',
          auto_schedule: true,
          recipients: ['risk@company.com'],
          created_by: 'system',
          created_at: new Date(),
          is_system_template: true
        },
        {
          id: 'template-weekly-comprehensive',
          name: 'Weekly Comprehensive Report',
          description: 'Weekly risk report with stress tests and scenario analysis',
          report_type: 'weekly',
          sections: [
            { id: 'summary', name: 'Executive Summary', type: 'summary', enabled: true },
            { id: 'var-metrics', name: 'VaR Metrics', type: 'metrics', enabled: true },
            { id: 'stress-tests', name: 'Stress Test Results', type: 'analysis', enabled: true },
            { id: 'scenario-analysis', name: 'Scenario Analysis', type: 'analysis', enabled: true },
            { id: 'performance-attribution', name: 'Performance Attribution', type: 'chart', enabled: true }
          ],
          default_format: 'pdf',
          frequency: 'weekly',
          auto_schedule: true,
          recipients: ['risk@company.com', 'management@company.com'],
          created_by: 'system',
          created_at: new Date(),
          is_system_template: true
        }
      ];

      updateState({
        templates: mockTemplates,
        error: null,
        loading: { ...state.loading, templates: false }
      });

      return mockTemplates;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, templates: false }
        });
      }
      throw error;
    }
  }, [updateState, state.loading]);

  const fetchSchedules = useCallback(async (silent = false) => {
    if (!silent) {
      updateState({ loading: { ...state.loading, schedules: true } });
    }

    try {
      // Mock API call - would be replaced with actual Sprint 3 backend endpoint
      const mockSchedules: ReportSchedule[] = [
        {
          id: 'schedule-1',
          template_id: 'template-daily-standard',
          portfolio_id: portfolioId,
          frequency: 'daily',
          time_of_day: '08:00',
          enabled: true,
          next_run: new Date(Date.now() + 86400000), // Tomorrow
          last_run: new Date(Date.now() - 86400000), // Yesterday
          recipients: ['risk@company.com'],
          delivery_method: 'email'
        }
      ];

      updateState({
        schedules: mockSchedules,
        error: null,
        loading: { ...state.loading, schedules: false }
      });

      return mockSchedules;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, schedules: false }
        });
      }
      throw error;
    }
  }, [portfolioId, updateState, state.loading]);

  const generateReport = useCallback(async (params: GenerateReportParams) => {
    const reportId = `report-${Date.now()}`;
    
    updateState({ 
      loading: { ...state.loading, generating: true },
      activeGenerations: new Set([...state.activeGenerations, reportId])
    });

    try {
      // Mock API call - would be replaced with actual Sprint 3 backend endpoint
      const reportData = await riskService.generateRiskReport(
        portfolioId,
        params.report_type,
        {
          include_stress_tests: params.include_stress_tests,
          include_scenarios: params.include_scenarios,
          format: params.format,
          date_range: {
            start: params.period_start,
            end: params.period_end
          }
        }
      );

      const newReport: RiskReport = {
        id: reportId,
        portfolio_id: portfolioId,
        report_type: params.report_type,
        format: params.format,
        generated_at: new Date(),
        period_start: params.period_start,
        period_end: params.period_end,
        status: 'generating',
        sections_included: params.sections || [],
        metadata: {}
      };

      updateState({
        reports: [newReport, ...state.reports],
        error: null,
        loading: { ...state.loading, generating: false }
      });

      // Start polling for completion
      pollReportStatus(reportId);

      return newReport;
    } catch (error) {
      if (error instanceof Error) {
        updateState({
          error: error.message,
          loading: { ...state.loading, generating: false },
          activeGenerations: new Set([...state.activeGenerations].filter(id => id !== reportId))
        });
      }
      throw error;
    }
  }, [portfolioId, state.loading, state.activeGenerations, state.reports, updateState]);

  const pollReportStatus = useCallback((reportId: string) => {
    const poll = async () => {
      try {
        // Mock status check - would be replaced with actual API call
        const isCompleted = Math.random() > 0.3; // Simulate completion
        
        if (isCompleted) {
          updateState({
            reports: state.reports.map(report => 
              report.id === reportId 
                ? { 
                    ...report, 
                    status: 'completed',
                    file_url: `/api/v1/risk/reports/${reportId}/download`,
                    file_size_bytes: Math.floor(Math.random() * 5000000) + 1000000,
                    metadata: {
                      ...report.metadata,
                      generation_time_seconds: Math.floor(Math.random() * 60) + 5
                    }
                  }
                : report
            ),
            activeGenerations: new Set([...state.activeGenerations].filter(id => id !== reportId))
          });
        } else {
          // Continue polling
          pollTimeoutRef.current = setTimeout(poll, 2000);
        }
      } catch (error) {
        console.error('Error polling report status:', error);
        updateState({
          reports: state.reports.map(report => 
            report.id === reportId 
              ? { 
                  ...report, 
                  status: 'failed',
                  error_message: error instanceof Error ? error.message : 'Unknown error'
                }
              : report
          ),
          activeGenerations: new Set([...state.activeGenerations].filter(id => id !== reportId))
        });
      }
    };

    pollTimeoutRef.current = setTimeout(poll, 2000);
  }, [state.reports, state.activeGenerations, updateState]);

  const downloadReport = useCallback(async (reportId: string) => {
    const report = state.reports.find(r => r.id === reportId);
    if (!report || !report.file_url) {
      throw new Error('Report not found or not ready for download');
    }

    try {
      const response = await fetch(report.file_url);
      if (!response.ok) {
        throw new Error('Failed to download report');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `risk-report-${report.id}.${report.format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      return true;
    } catch (error) {
      if (error instanceof Error) {
        updateState({ error: error.message });
      }
      throw error;
    }
  }, [state.reports, updateState]);

  const deleteReport = useCallback(async (reportId: string) => {
    try {
      // Mock API call - would be replaced with actual delete endpoint
      updateState({
        reports: state.reports.filter(report => report.id !== reportId),
        error: null
      });

      return true;
    } catch (error) {
      if (error instanceof Error) {
        updateState({ error: error.message });
      }
      throw error;
    }
  }, [state.reports, updateState]);

  const createSchedule = useCallback(async (scheduleData: Omit<ReportSchedule, 'id' | 'next_run'>) => {
    try {
      const newSchedule: ReportSchedule = {
        ...scheduleData,
        id: `schedule-${Date.now()}`,
        next_run: new Date(Date.now() + 86400000) // Tomorrow
      };

      updateState({
        schedules: [...state.schedules, newSchedule],
        error: null
      });

      return newSchedule;
    } catch (error) {
      if (error instanceof Error) {
        updateState({ error: error.message });
      }
      throw error;
    }
  }, [state.schedules, updateState]);

  const clearError = useCallback(() => {
    updateState({ error: null });
  }, [updateState]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      const refresh = () => {
        Promise.all([
          fetchReports(true),
          fetchSchedules(true)
        ]).catch(console.error);
      };

      refreshTimeoutRef.current = setTimeout(function tick() {
        refresh();
        refreshTimeoutRef.current = setTimeout(tick, refreshInterval);
      }, refreshInterval);

      return () => {
        if (refreshTimeoutRef.current) {
          clearTimeout(refreshTimeoutRef.current);
        }
      };
    }
  }, [autoRefresh, refreshInterval, fetchReports, fetchSchedules]);

  // Initial fetch
  useEffect(() => {
    Promise.all([
      fetchReports(),
      fetchTemplates(),
      fetchSchedules()
    ]).catch(console.error);

    return () => {
      if (refreshTimeoutRef.current) {
        clearTimeout(refreshTimeoutRef.current);
      }
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
    };
  }, [portfolioId, fetchReports, fetchTemplates, fetchSchedules]);

  // Computed values
  const recentReports = state.reports
    .filter(r => r.status === 'completed')
    .sort((a, b) => b.generated_at.getTime() - a.generated_at.getTime())
    .slice(0, 10);

  const completedReports = state.reports.filter(r => r.status === 'completed');
  const failedReports = state.reports.filter(r => r.status === 'failed');
  const generatingReports = state.reports.filter(r => r.status === 'generating');

  const activeSchedules = state.schedules.filter(s => s.enabled);

  const reportsByType = state.reports.reduce((acc, report) => {
    if (!acc[report.report_type]) {
      acc[report.report_type] = [];
    }
    acc[report.report_type].push(report);
    return acc;
  }, {} as Record<string, RiskReport[]>);

  return {
    // State
    ...state,
    
    // Computed values
    recentReports,
    completedReports,
    failedReports,
    generatingReports,
    activeSchedules,
    reportsByType,
    
    // Actions
    fetchReports,
    fetchTemplates,
    fetchSchedules,
    generateReport,
    downloadReport,
    deleteReport,
    createSchedule,
    clearError,
    
    // Utilities
    refresh: () => Promise.all([
      fetchReports(),
      fetchTemplates(),
      fetchSchedules()
    ]),
    
    isGenerating: (reportId?: string) => 
      reportId ? state.activeGenerations.has(reportId) : state.activeGenerations.size > 0,
      
    getTemplate: (templateId: string) => 
      state.templates.find(t => t.id === templateId),
      
    getSchedule: (scheduleId: string) => 
      state.schedules.find(s => s.id === scheduleId)
  };
};

export default useRiskReporting;