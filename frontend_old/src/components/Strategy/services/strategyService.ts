import axios, { AxiosResponse } from 'axios';
import {
  StrategyTemplate,
  TemplateResponse,
  StrategyConfig,
  ConfigureRequest,
  ConfigureResponse,
  DeployRequest,
  DeployResponse,
  ControlRequest,
  ControlResponse,
  StatusResponse,
  StrategySearchFilters,
  StrategyVersion,
  VersionComparisonResult,
  ConfigurationChange,
  ConfigurationAudit,
  PerformanceMetrics,
  RollbackPlan,
  RollbackValidation,
  RollbackProgress
} from '../types/strategyTypes';

const API_BASE = 'http://localhost:8080/api/v1';

class StrategyService {
  private async apiCall<T>(url: string, options?: any): Promise<T> {
    try {
      const response: AxiosResponse<T> = await axios({
        url: `${API_BASE}${url}`,
        timeout: 10000,
        ...options
      });
      return response.data;
    } catch (error: any) {
      console.error(`Strategy API Error [${url}]:`, error);
      throw new Error(
        error.response?.data?.error?.message || 
        error.message || 
        'Strategy service error'
      );
    }
  }

  // Strategy Templates
  async getTemplates(): Promise<TemplateResponse> {
    return this.apiCall<TemplateResponse>('/strategies/templates');
  }

  async getTemplate(templateId: string): Promise<StrategyTemplate> {
    return this.apiCall<StrategyTemplate>(`/strategies/templates/${templateId}`);
  }

  async searchTemplates(filters: StrategySearchFilters): Promise<TemplateResponse> {
    const params = new URLSearchParams();
    if (filters.category) params.append('category', filters.category);
    if (filters.search_text) params.append('search', filters.search_text);
    if (filters.tags?.length) {
      filters.tags.forEach(tag => params.append('tags', tag));
    }

    return this.apiCall<TemplateResponse>(`/strategies/templates?${params}`);
  }

  // Strategy Configuration
  async createConfiguration(request: ConfigureRequest): Promise<ConfigureResponse> {
    return this.apiCall<ConfigureResponse>('/strategies/configure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: request
    });
  }

  async updateConfiguration(
    strategyId: string, 
    updates: Partial<ConfigureRequest>
  ): Promise<ConfigureResponse> {
    return this.apiCall<ConfigureResponse>(`/strategies/configure/${strategyId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      data: updates
    });
  }

  async getConfiguration(strategyId: string): Promise<StrategyConfig> {
    return this.apiCall<StrategyConfig>(`/strategies/configure/${strategyId}`);
  }

  async deleteConfiguration(strategyId: string): Promise<void> {
    return this.apiCall<void>(`/strategies/configure/${strategyId}`, {
      method: 'DELETE'
    });
  }

  async listConfigurations(): Promise<StrategyConfig[]> {
    return this.apiCall<StrategyConfig[]>('/strategies/configurations');
  }

  // Strategy Deployment
  async deployStrategy(request: DeployRequest): Promise<DeployResponse> {
    return this.apiCall<DeployResponse>('/strategies/deploy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: request
    });
  }

  async getDeploymentStatus(deploymentId: string): Promise<StatusResponse> {
    return this.apiCall<StatusResponse>(`/strategies/deployments/${deploymentId}/status`);
  }

  // Strategy Control
  async controlStrategy(
    strategyId: string, 
    request: ControlRequest
  ): Promise<ControlResponse> {
    return this.apiCall<ControlResponse>(`/strategies/${strategyId}/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: request
    });
  }

  async getStrategyStatus(strategyId: string): Promise<StatusResponse> {
    return this.apiCall<StatusResponse>(`/strategies/status/${strategyId}`);
  }

  // Validation
  async validateParameters(
    templateId: string, 
    parameters: Record<string, any>
  ): Promise<{ valid: boolean; errors: string[] }> {
    return this.apiCall('/strategies/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { template_id: templateId, parameters }
    });
  }

  // Utility methods
  async getAvailableInstruments(): Promise<string[]> {
    return this.apiCall<string[]>('/instruments/available');
  }

  async getAvailableTimeframes(): Promise<string[]> {
    return this.apiCall<string[]>('/timeframes/available');
  }

  async getAvailableVenues(): Promise<string[]> {
    return this.apiCall<string[]>('/venues/available');
  }

  // Health check for strategy service
  async healthCheck(): Promise<{ status: string; timestamp: Date }> {
    return this.apiCall<{ status: string; timestamp: Date }>('/strategies/health');
  }

  // Version Control Methods
  async getVersionHistory(strategyId: string): Promise<{ versions: StrategyVersion[] }> {
    return this.apiCall<{ versions: StrategyVersion[] }>(`/strategies/${strategyId}/versions`);
  }

  async createVersion(
    strategyId: string,
    changeSummary: string,
    saveCurrentConfig: boolean = true
  ): Promise<StrategyVersion> {
    return this.apiCall<StrategyVersion>(`/strategies/${strategyId}/versions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { change_summary: changeSummary, save_current_config: saveCurrentConfig }
    });
  }

  async getVersion(strategyId: string, versionId: string): Promise<StrategyVersion> {
    return this.apiCall<StrategyVersion>(`/strategies/${strategyId}/versions/${versionId}`);
  }

  async compareVersions(
    strategyId: string,
    version1Id: string,
    version2Id: string
  ): Promise<VersionComparisonResult> {
    return this.apiCall<VersionComparisonResult>(
      `/strategies/${strategyId}/versions/compare?version1=${version1Id}&version2=${version2Id}`
    );
  }

  async rollbackToVersion(
    strategyId: string,
    versionId: string
  ): Promise<{ success: boolean; error?: string; new_version?: number }> {
    return this.apiCall<{ success: boolean; error?: string; new_version?: number }>(
      `/strategies/${strategyId}/rollback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        data: { version_id: versionId }
      }
    );
  }

  // Configuration History Methods
  async getConfigurationHistory(strategyId: string): Promise<{ changes: ConfigurationChange[] }> {
    return this.apiCall<{ changes: ConfigurationChange[] }>(`/strategies/${strategyId}/history`);
  }

  async getConfigurationAudit(strategyId: string): Promise<{ audit_entries: ConfigurationAudit[] }> {
    return this.apiCall<{ audit_entries: ConfigurationAudit[] }>(`/strategies/${strategyId}/audit`);
  }

  async getPerformanceHistory(strategyId: string): Promise<{ metrics: PerformanceMetrics[] }> {
    return this.apiCall<{ metrics: PerformanceMetrics[] }>(`/strategies/${strategyId}/performance/history`);
  }

  // Rollback System Methods
  async generateRollbackPlan(
    strategyId: string,
    fromVersion: number,
    toVersion: number
  ): Promise<RollbackPlan> {
    return this.apiCall<RollbackPlan>(`/strategies/${strategyId}/rollback/plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { from_version: fromVersion, to_version: toVersion }
    });
  }

  async validateRollback(
    strategyId: string,
    targetVersion: number,
    rollbackPlan: RollbackPlan
  ): Promise<RollbackValidation> {
    return this.apiCall<RollbackValidation>(`/strategies/${strategyId}/rollback/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { target_version: targetVersion, rollback_plan: rollbackPlan }
    });
  }

  async executeRollback(
    strategyId: string,
    targetVersion: number,
    rollbackSettings: any,
    progressCallback?: (progress: RollbackProgress) => void
  ): Promise<{ success: boolean; error?: string }> {
    // For real-time progress, we would typically use WebSockets or Server-Sent Events
    // This is a simplified version using polling
    const result = await this.apiCall<{ rollback_id: string }>(`/strategies/${strategyId}/rollback/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { target_version: targetVersion, settings: rollbackSettings }
    });

    // Poll for progress if callback provided
    if (progressCallback && result.rollback_id) {
      const pollProgress = async () => {
        try {
          const progress = await this.getRollbackProgress(result.rollback_id);
          progressCallback(progress);
          
          if (progress.status === 'running' || progress.status === 'initializing') {
            setTimeout(pollProgress, 1000); // Poll every second
          }
        } catch (error) {
          console.error('Error polling rollback progress:', error);
        }
      };
      setTimeout(pollProgress, 1000);
    }

    return { success: true };
  }

  async getRollbackProgress(rollbackId: string): Promise<RollbackProgress> {
    return this.apiCall<RollbackProgress>(`/strategies/rollback/${rollbackId}/progress`);
  }

  // Configuration Snapshots
  async createConfigurationSnapshot(
    strategyId: string,
    description?: string
  ): Promise<{ snapshot_id: string }> {
    return this.apiCall<{ snapshot_id: string }>(`/strategies/${strategyId}/snapshots`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: { description }
    });
  }

  async restoreFromSnapshot(
    strategyId: string,
    snapshotId: string
  ): Promise<{ success: boolean; error?: string }> {
    return this.apiCall<{ success: boolean; error?: string }>(`/strategies/${strategyId}/snapshots/${snapshotId}/restore`, {
      method: 'POST'
    });
  }

  // Performance Tracking for Version Control
  async getVersionPerformanceComparison(
    strategyId: string,
    version1: number,
    version2: number
  ): Promise<{
    version1_metrics: PerformanceMetrics;
    version2_metrics: PerformanceMetrics;
    comparison: any;
  }> {
    return this.apiCall<{
      version1_metrics: PerformanceMetrics;
      version2_metrics: PerformanceMetrics;
      comparison: any;
    }>(`/strategies/${strategyId}/performance/compare?v1=${version1}&v2=${version2}`);
  }
}

export const strategyService = new StrategyService();
export default strategyService;