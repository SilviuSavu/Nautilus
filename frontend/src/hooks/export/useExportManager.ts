/**
 * Story 5.3: Export Manager Hook
 * React hook for managing data exports and export history
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { exportService } from '../../services/export/ExportService';
import {
  ExportRequest,
  ExportRequestResponse,
  ExportStatusResponse,
  ExportDownloadResponse,
  ExportHistory,
  SupportedFormatsResponse,
  AvailableFieldsResponse,
  DataSource,
  ExportStatus,
  UseExportManagerConfig,
  UseExportManagerReturn
} from '../../types/export';

const DEFAULT_CONFIG: UseExportManagerConfig = {
  autoRefresh: true,
  refreshInterval: 5000 // 5 seconds
};

export const useExportManager = (
  config: UseExportManagerConfig = {}
): UseExportManagerReturn => {
  const configWithDefaults = { ...DEFAULT_CONFIG, ...config };
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // State
  const [exports, setExports] = useState<ExportHistory[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Error handling helper
  const handleError = useCallback((error: Error, operation: string) => {
    console.error(`Error in ${operation}:`, error);
    setError(error.message);
    setLoading(false);
  }, []);

  // Clear error helper
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Create export request
  const createExport = useCallback(async (request: ExportRequest): Promise<ExportRequestResponse> => {
    try {
      setLoading(true);
      clearError();

      const response = await exportService.createExport(request);
      
      // Add the new export to the list with pending status
      const newExport: ExportHistory = {
        id: response.export_id,
        type: request.type,
        data_source: request.data_source,
        status: ExportStatus.PENDING,
        progress: 0,
        created_at: new Date(),
      };

      setExports(prev => [newExport, ...prev]);
      setLoading(false);
      
      return response;
    } catch (error) {
      handleError(error as Error, 'createExport');
      throw error;
    }
  }, [handleError, clearError]);

  // Get export status
  const getExportStatus = useCallback(async (exportId: string): Promise<ExportStatusResponse> => {
    try {
      clearError();
      return await exportService.getExportStatus(exportId);
    } catch (error) {
      handleError(error as Error, 'getExportStatus');
      throw error;
    }
  }, [handleError, clearError]);

  // Download export
  const downloadExport = useCallback(async (exportId: string): Promise<ExportDownloadResponse> => {
    try {
      clearError();
      const response = await exportService.downloadExport(exportId);
      
      // Create a temporary link to download the file
      const link = document.createElement('a');
      link.href = response.download_url;
      link.download = response.file_name;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      return response;
    } catch (error) {
      handleError(error as Error, 'downloadExport');
      throw error;
    }
  }, [handleError, clearError]);

  // Delete export
  const deleteExport = useCallback(async (exportId: string): Promise<void> => {
    try {
      clearError();
      await exportService.deleteExport(exportId);
      
      // Remove from local state
      setExports(prev => prev.filter(exp => exp.id !== exportId));
    } catch (error) {
      handleError(error as Error, 'deleteExport');
      throw error;
    }
  }, [handleError, clearError]);

  // Refresh exports list
  const refreshExports = useCallback(async () => {
    try {
      setLoading(true);
      clearError();

      // The updated getExportHistory method now handles fallbacks internally
      const response = await exportService.getExportHistory();
      setExports(response.exports);

      setLoading(false);
    } catch (error) {
      handleError(error as Error, 'refreshExports');
    }
  }, [handleError, clearError]);

  // Get supported formats
  const getSupportedFormats = useCallback(async (): Promise<SupportedFormatsResponse> => {
    try {
      clearError();
      return await exportService.getSupportedFormats();
    } catch (error) {
      // Return fallback formats if API fails
      return {
        formats: [
          { type: 'csv', name: 'CSV (Comma-Separated Values)', extension: '.csv', supports_compression: true },
          { type: 'json', name: 'JSON (JavaScript Object Notation)', extension: '.json', supports_compression: true },
          { type: 'excel', name: 'Excel Spreadsheet', extension: '.xlsx', supports_compression: false },
          { type: 'pdf', name: 'PDF Document', extension: '.pdf', supports_compression: false }
        ]
      };
    }
  }, [clearError]);

  // Get available fields
  const getAvailableFields = useCallback(async (dataSource: DataSource): Promise<AvailableFieldsResponse> => {
    try {
      clearError();
      return await exportService.getAvailableFields(dataSource);
    } catch (error) {
      // Return fallback fields if API fails
      const fieldMappings: Record<DataSource, string[]> = {
        [DataSource.TRADES]: ['id', 'timestamp', 'symbol', 'side', 'quantity', 'price', 'value', 'commission', 'pnl'],
        [DataSource.PERFORMANCE]: ['timestamp', 'total_pnl', 'unrealized_pnl', 'win_rate', 'sharpe_ratio', 'max_drawdown'],
        [DataSource.POSITIONS]: ['symbol', 'quantity', 'market_value', 'unrealized_pnl', 'cost_basis'],
        [DataSource.ORDERS]: ['id', 'timestamp', 'symbol', 'side', 'quantity', 'price', 'status'],
        [DataSource.SYSTEM_METRICS]: ['timestamp', 'cpu_usage', 'memory_usage', 'latency_avg', 'venue']
      };

      return {
        data_source: dataSource,
        available_fields: fieldMappings[dataSource] || []
      };
    }
  }, [clearError]);

  // Update export status for active exports
  const updateExportStatuses = useCallback(async () => {
    const activeExports = exports.filter(exp => 
      exp.status === ExportStatus.PENDING || exp.status === ExportStatus.PROCESSING
    );

    if (activeExports.length === 0) return;

    try {
      const statusPromises = activeExports.map(exp => 
        exportService.getExportStatus(exp.id).catch(error => {
          console.warn(`Failed to get status for export ${exp.id}:`, error);
          return null;
        })
      );

      const statuses = await Promise.all(statusPromises);

      setExports(prev => prev.map(exp => {
        const statusIndex = activeExports.findIndex(active => active.id === exp.id);
        if (statusIndex >= 0 && statuses[statusIndex]) {
          const status = statuses[statusIndex]!;
          return {
            ...exp,
            status: status.status,
            progress: status.progress,
            download_url: status.download_url,
            completed_at: status.completed_at
          };
        }
        return exp;
      }));
    } catch (error) {
      console.error('Failed to update export statuses:', error);
    }
  }, [exports]);

  // Start auto refresh
  const startAutoRefresh = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    intervalRef.current = setInterval(() => {
      updateExportStatuses();
    }, configWithDefaults.refreshInterval);
  }, [updateExportStatuses, configWithDefaults.refreshInterval]);

  // Stop auto refresh
  const stopAutoRefresh = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // Initialize and auto-refresh
  useEffect(() => {
    // Initial data load
    refreshExports();

    // Start auto-refresh if enabled
    if (configWithDefaults.autoRefresh) {
      startAutoRefresh();
    }

    // Cleanup on unmount
    return () => {
      stopAutoRefresh();
    };
  }, []); // Empty dependency array for mount/unmount only

  // Update export statuses when exports change
  useEffect(() => {
    if (configWithDefaults.autoRefresh && exports.length > 0) {
      const hasActiveExports = exports.some(exp => 
        exp.status === ExportStatus.PENDING || exp.status === ExportStatus.PROCESSING
      );

      if (hasActiveExports && !intervalRef.current) {
        startAutoRefresh();
      } else if (!hasActiveExports && intervalRef.current) {
        stopAutoRefresh();
      }
    }
  }, [exports, configWithDefaults.autoRefresh, startAutoRefresh, stopAutoRefresh]);

  return {
    // State
    exports,
    loading,
    error,

    // Actions
    createExport,
    getExportStatus,
    downloadExport,
    deleteExport,
    refreshExports,

    // Utility
    getSupportedFormats,
    getAvailableFields
  };
};

export default useExportManager;