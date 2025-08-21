/**
 * Story 5.3: Data Export and Reporting TypeScript Types
 * Type definitions for export, reporting, and API integration functionality
 */

// Core enums
export enum ExportType {
  CSV = 'csv',
  JSON = 'json',
  EXCEL = 'excel',
  PDF = 'pdf'
}

export enum DataSource {
  TRADES = 'trades',
  POSITIONS = 'positions',
  PERFORMANCE = 'performance',
  ORDERS = 'orders',
  SYSTEM_METRICS = 'system_metrics'
}

export enum ExportStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

export enum ReportType {
  PERFORMANCE = 'performance',
  COMPLIANCE = 'compliance',
  RISK = 'risk',
  CUSTOM = 'custom'
}

export enum ReportFormat {
  PDF = 'pdf',
  EXCEL = 'excel',
  HTML = 'html'
}

export enum ScheduleFrequency {
  DAILY = 'daily',
  WEEKLY = 'weekly',
  MONTHLY = 'monthly',
  QUARTERLY = 'quarterly'
}

export enum AuthenticationType {
  API_KEY = 'api_key',
  OAUTH = 'oauth',
  BASIC = 'basic'
}

export enum IntegrationStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  ERROR = 'error'
}

// Core interfaces
export interface DateRange {
  start_date: Date;
  end_date: Date;
}

export interface ExportFilters {
  date_range: DateRange;
  symbols?: string[];
  accounts?: string[];
  strategies?: string[];
  venues?: string[];
}

export interface ExportOptions {
  include_headers: boolean;
  compression: boolean;
  precision: number;
  timezone: string;
  currency: string;
}

export interface ExportRequest {
  id?: string;
  type: ExportType;
  data_source: DataSource;
  filters: ExportFilters;
  fields: string[];
  options: ExportOptions;
  status: ExportStatus;
  progress: number;
  download_url?: string;
  created_at?: Date;
  completed_at?: Date;
}

export interface ExportHistory {
  id: string;
  type: ExportType;
  data_source: DataSource;
  status: ExportStatus;
  progress: number;
  created_at: Date;
  completed_at?: Date;
  download_url?: string;
}

// Report interfaces
export interface ReportSchedule {
  frequency: ScheduleFrequency;
  time: string;
  timezone: string;
  recipients: string[];
}

export interface ReportSection {
  id: string;
  name: string;
  type: string;
  configuration: Record<string, any>;
}

export interface ReportParameter {
  name: string;
  type: string;
  default_value: any;
  required: boolean;
}

export interface ReportTemplate {
  id?: string;
  name: string;
  description: string;
  type: ReportType;
  format: ReportFormat;
  schedule?: ReportSchedule;
  sections: ReportSection[];
  parameters: ReportParameter[];
  created_at?: Date;
  updated_at?: Date;
}

// API Integration interfaces
export interface FieldMapping {
  source_field: string;
  target_field: string;
  transformation?: string;
}

export interface ApiIntegration {
  id?: string;
  name: string;
  endpoint: string;
  authentication: Record<string, any>;
  data_mapping: FieldMapping[];
  schedule?: Record<string, any>;
  status: IntegrationStatus;
  created_at?: Date;
  last_sync?: Date;
}

// API Response interfaces
export interface ExportRequestResponse {
  export_id: string;
  status: ExportStatus;
  message: string;
}

export interface ExportStatusResponse {
  export_id: string;
  status: ExportStatus;
  progress: number;
  created_at: Date;
  completed_at?: Date;
  download_url?: string;
}

export interface ExportDownloadResponse {
  export_id: string;
  file_name: string;
  file_size: string;
  download_url: string;
  expires_at: string;
}

export interface ExportHistoryResponse {
  exports: ExportHistory[];
  total_count: number;
}

export interface ReportTemplatesResponse {
  templates: ReportTemplate[];
  total_count: number;
}

export interface ReportGenerationResponse {
  report_id: string;
  template_id: string;
  status: string;
  download_url: string;
  generated_at: string;
}

export interface ScheduledReportsResponse {
  scheduled_reports: ReportTemplate[];
  total_count: number;
}

export interface ApiIntegrationsResponse {
  integrations: ApiIntegration[];
  total_count: number;
}

export interface IntegrationSyncResponse {
  integration_id: string;
  status: string;
  last_sync: string;
}

export interface IntegrationStatusResponse {
  integration_id: string;
  name: string;
  status: IntegrationStatus;
  last_sync?: Date;
  endpoint: string;
  created_at?: Date;
}

export interface SupportedFormatsResponse {
  formats: {
    type: string;
    name: string;
    extension: string;
    supports_compression: boolean;
  }[];
}

export interface AvailableFieldsResponse {
  data_source: DataSource;
  available_fields: string[];
}

// Configuration interfaces
export interface ExportConfig {
  defaultType: ExportType;
  defaultOptions: ExportOptions;
  maxFileSize: number;
  allowedSources: DataSource[];
  compressionEnabled: boolean;
}

export interface ReportConfig {
  templateDirectory: string;
  maxSections: number;
  supportedFormats: ReportFormat[];
  schedulingEnabled: boolean;
  distributionEnabled: boolean;
}

export interface IntegrationConfig {
  maxIntegrations: number;
  supportedAuthTypes: AuthenticationType[];
  rateLimitEnabled: boolean;
  webhookEnabled: boolean;
}

// UI State interfaces
export interface ExportWizardState {
  step: number;
  exportRequest: Partial<ExportRequest>;
  validation: Record<string, string>;
  isValid: boolean;
}

export interface ReportBuilderState {
  template: Partial<ReportTemplate>;
  selectedSections: string[];
  parameters: Record<string, any>;
  previewData: any;
}

export interface IntegrationSetupState {
  integration: Partial<ApiIntegration>;
  testResult?: {
    success: boolean;
    message: string;
    data?: any;
  };
  fieldMappings: FieldMapping[];
}

// Hook interfaces
export interface UseExportManagerConfig {
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export interface UseExportManagerReturn {
  // State
  exports: ExportHistory[];
  loading: boolean;
  error: string | null;
  
  // Actions
  createExport: (request: ExportRequest) => Promise<ExportRequestResponse>;
  getExportStatus: (exportId: string) => Promise<ExportStatusResponse>;
  downloadExport: (exportId: string) => Promise<ExportDownloadResponse>;
  deleteExport: (exportId: string) => Promise<void>;
  refreshExports: () => Promise<void>;
  
  // Utility
  getSupportedFormats: () => Promise<SupportedFormatsResponse>;
  getAvailableFields: (dataSource: DataSource) => Promise<AvailableFieldsResponse>;
}

export interface UseReportGeneratorConfig {
  autoLoadTemplates?: boolean;
}

export interface UseReportGeneratorReturn {
  // State
  templates: ReportTemplate[];
  scheduledReports: ReportTemplate[];
  loading: boolean;
  error: string | null;
  
  // Actions
  createTemplate: (template: ReportTemplate) => Promise<{ template_id: string; message: string }>;
  generateReport: (templateId: string, parameters: Record<string, any>, deliveryOptions: Record<string, any>) => Promise<ReportGenerationResponse>;
  refreshTemplates: () => Promise<void>;
  refreshScheduledReports: () => Promise<void>;
}

export interface UseIntegrationsConfig {
  autoSync?: boolean;
  syncInterval?: number;
}

export interface UseIntegrationsReturn {
  // State
  integrations: ApiIntegration[];
  loading: boolean;
  error: string | null;
  
  // Actions
  createIntegration: (integration: ApiIntegration) => Promise<{ integration_id: string; message: string }>;
  syncIntegration: (integrationId: string) => Promise<IntegrationSyncResponse>;
  getIntegrationStatus: (integrationId: string) => Promise<IntegrationStatusResponse>;
  refreshIntegrations: () => Promise<void>;
}