export interface DataGap {
  id: string
  start: Date
  end: Date
  severity: 'low' | 'medium' | 'high'
  reason?: string
  detectedAt: Date
  filledAt?: Date
}

export interface QualityMetrics {
  completeness: number
  accuracy: number
  timeliness: number
  consistency: number
  overall: number
  lastUpdated: Date
}

export interface QualityReport {
  instrumentId: string
  metrics: QualityMetrics
  anomalies: Anomaly[]
  validationResults: ValidationResult[]
  generatedAt: Date
}

export interface Anomaly {
  id: string
  instrumentId: string
  timestamp: Date
  type: 'spike' | 'gap' | 'inconsistency' | 'outlier'
  severity: 'low' | 'medium' | 'high'
  description: string
  value?: number
  expectedValue?: number
}

export interface ValidationResult {
  checkName: string
  passed: boolean
  message: string
  timestamp: Date
}

export interface DataSourceInfo {
  id: string
  name: string
  type: 'realtime' | 'historical' | 'hybrid'
  status: 'active' | 'inactive' | 'error'
  priority: number
  latency?: number
  throughput?: number
  errorRate?: number
  lastUpdate: Date
}

export interface VenueMetadata {
  id: string
  name: string
  timezone: string
  tradingHours: {
    open: string
    close: string
    breaks?: Array<{ start: string; end: string }>
  }
  instrumentCount: number
  dataTypes: Array<'tick' | 'quote' | 'bar'>
  qualityScore: number
}

export interface InstrumentMetadata {
  instrumentId: string
  venue: string
  symbol: string
  description?: string
  assetClass: string
  currency: string
  dataType: 'tick' | 'quote' | 'bar'
  timeframes: string[]
  dateRange: {
    start: Date
    end: Date
  }
  recordCount: number
  qualityScore: number
  gaps: DataGap[]
  lastUpdated: Date
  fileSize?: number
  compressionRatio?: number
}

export interface DataCatalog {
  instruments: InstrumentMetadata[]
  venues: VenueMetadata[]
  dataSources: DataSourceInfo[]
  qualityMetrics: QualityMetrics
  lastUpdated: Date
  totalInstruments: number
  totalRecords: number
  storageSize: number
}

export interface DataFeedStatus {
  feedId: string
  source: string
  instrumentId?: string
  status: 'connected' | 'disconnected' | 'degraded' | 'reconnecting'
  latency: number
  throughput: number
  lastUpdate: Date
  errorCount: number
  qualityScore: number
  subscriptionCount: number
  bandwidth?: number
}

export interface DataExportRequest {
  instrumentIds: string[]
  format: 'parquet' | 'csv' | 'json' | 'nautilus'
  dateRange: {
    start: string
    end: string
  }
  timeframes?: string[]
  compression?: boolean
  includeMetadata?: boolean
  maxRecords?: number
}

export interface ExportResult {
  exportId: string
  success: boolean
  filePath?: string
  downloadUrl?: string
  recordCount?: number
  fileSize?: number
  format: string
  createdAt: Date
  completedAt?: Date
  error?: string
}

export interface ImportRequest {
  filePath: string
  format: 'parquet' | 'csv' | 'json'
  instrumentId?: string
  venue?: string
  dataType?: 'tick' | 'quote' | 'bar'
  validateData?: boolean
  overwrite?: boolean
}

export interface ImportResult {
  importId: string
  success: boolean
  recordCount?: number
  instrumentId?: string
  validationResults?: ValidationResult[]
  warnings?: string[]
  error?: string
  processedAt: Date
}

export interface IntegrityReport {
  instrumentId: string
  totalRecords: number
  validRecords: number
  invalidRecords: number
  duplicateRecords: number
  missingFields: string[]
  dataRangeIssues: string[]
  crossValidationResults: Array<{
    source: string
    matched: boolean
    discrepancy?: string
  }>
  generatedAt: Date
}

export interface CatalogSearchFilters {
  venue?: string
  assetClass?: string
  currency?: string
  dataType?: 'tick' | 'quote' | 'bar'
  timeframe?: string
  qualityThreshold?: number
  dateRange?: {
    start: string
    end: string
  }
  hasGaps?: boolean
  recordCountMin?: number
  recordCountMax?: number
}

export interface CatalogSearchResult {
  instruments: InstrumentMetadata[]
  totalCount: number
  pageSize: number
  currentPage: number
  filters: CatalogSearchFilters
}