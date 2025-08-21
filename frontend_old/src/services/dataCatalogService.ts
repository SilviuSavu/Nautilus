import axios from 'axios'
import {
  DataCatalog,
  InstrumentMetadata,
  DataGap,
  QualityReport,
  DataExportRequest,
  ExportResult,
  ImportRequest,
  ImportResult,
  DataFeedStatus,
  QualityMetrics,
  IntegrityReport,
  CatalogSearchFilters,
  CatalogSearchResult,
  Anomaly
} from '../types/dataCatalog'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export class DataCatalogService {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl
  }

  async getCatalog(): Promise<DataCatalog> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/v1/nautilus/data/catalog`)
      return this.transformCatalogResponse(response.data)
    } catch (error) {
      console.error('Failed to fetch data catalog:', error)
      throw new Error('Unable to retrieve data catalog')
    }
  }

  async getInstrumentData(instrumentId: string): Promise<InstrumentMetadata> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/v1/nautilus/data/catalog/instruments/${instrumentId}`)
      return this.transformInstrumentResponse(response.data)
    } catch (error) {
      console.error(`Failed to fetch instrument data for ${instrumentId}:`, error)
      throw new Error(`Unable to retrieve data for instrument ${instrumentId}`)
    }
  }

  async searchInstruments(filters: CatalogSearchFilters): Promise<CatalogSearchResult> {
    try {
      const params = this.buildSearchParams(filters)
      const response = await axios.get(`${this.baseUrl}/api/v1/nautilus/data/catalog/search`, { params })
      return response.data
    } catch (error) {
      console.error('Failed to search instruments:', error)
      throw new Error('Unable to search instruments')
    }
  }

  async analyzeDataGaps(instrumentId: string): Promise<DataGap[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/v1/nautilus/data/gaps/${instrumentId}`)
      return response.data.gaps.map(this.transformGapResponse)
    } catch (error) {
      console.error(`Failed to analyze data gaps for ${instrumentId}:`, error)
      throw new Error(`Unable to analyze data gaps for ${instrumentId}`)
    }
  }

  async validateDataQuality(instrumentId: string): Promise<QualityReport> {
    try {
      const response = await axios.post(`${this.baseUrl}/api/v1/nautilus/data/quality/validate`, {
        instrumentId
      })
      return this.transformQualityResponse(response.data)
    } catch (error) {
      console.error(`Failed to validate data quality for ${instrumentId}:`, error)
      throw new Error(`Unable to validate data quality for ${instrumentId}`)
    }
  }

  async getQualityMetrics(instrumentId: string): Promise<QualityMetrics> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/v1/nautilus/data/quality/${instrumentId}`)
      return response.data
    } catch (error) {
      console.error(`Failed to get quality metrics for ${instrumentId}:`, error)
      throw new Error(`Unable to get quality metrics for ${instrumentId}`)
    }
  }

  async exportData(request: DataExportRequest): Promise<ExportResult> {
    try {
      const response = await axios.post(`${this.baseUrl}/api/v1/nautilus/data/export`, request)
      return response.data
    } catch (error) {
      console.error('Failed to export data:', error)
      throw new Error('Unable to export data')
    }
  }

  async importData(request: ImportRequest): Promise<ImportResult> {
    try {
      const response = await axios.post(`${this.baseUrl}/api/v1/nautilus/data/import`, request)
      return response.data
    } catch (error) {
      console.error('Failed to import data:', error)
      throw new Error('Unable to import data')
    }
  }

  async getFeedStatuses(): Promise<DataFeedStatus[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/v1/nautilus/data/feeds/status`)
      return response.data.feeds || []
    } catch (error) {
      console.error('Failed to get feed statuses:', error)
      throw new Error('Unable to get feed statuses')
    }
  }

  async subscribeFeed(feedId: string, instrumentIds: string[]): Promise<boolean> {
    try {
      await axios.post(`${this.baseUrl}/api/v1/nautilus/data/feeds/subscribe`, {
        feedId,
        instrumentIds
      })
      return true
    } catch (error) {
      console.error(`Failed to subscribe to feed ${feedId}:`, error)
      throw new Error(`Unable to subscribe to feed ${feedId}`)
    }
  }

  async unsubscribeFeed(feedId: string, instrumentIds: string[]): Promise<boolean> {
    try {
      await axios.delete(`${this.baseUrl}/api/v1/nautilus/data/feeds/unsubscribe`, {
        data: { feedId, instrumentIds }
      })
      return true
    } catch (error) {
      console.error(`Failed to unsubscribe from feed ${feedId}:`, error)
      throw new Error(`Unable to unsubscribe from feed ${feedId}`)
    }
  }

  async getPipelineHealth(): Promise<{ status: string; details: Record<string, any> }> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/v1/nautilus/data/pipeline/health`)
      return response.data
    } catch (error) {
      console.error('Failed to get pipeline health:', error)
      throw new Error('Unable to get pipeline health status')
    }
  }

  async getIntegrityReport(instrumentId: string): Promise<IntegrityReport> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/v1/nautilus/data/integrity/${instrumentId}`)
      return response.data
    } catch (error) {
      console.error(`Failed to get integrity report for ${instrumentId}:`, error)
      throw new Error(`Unable to get integrity report for ${instrumentId}`)
    }
  }

  async detectAnomalies(instrumentId: string, dateRange?: { start: string; end: string }): Promise<Anomaly[]> {
    try {
      const params = dateRange ? { start: dateRange.start, end: dateRange.end } : {}
      const response = await axios.get(`${this.baseUrl}/api/v1/nautilus/data/anomalies/${instrumentId}`, { params })
      return response.data.anomalies || []
    } catch (error) {
      console.error(`Failed to detect anomalies for ${instrumentId}:`, error)
      throw new Error(`Unable to detect anomalies for ${instrumentId}`)
    }
  }

  // Docker integration methods for Nautilus operations
  async executeNautilusCatalogCommand(command: string): Promise<any> {
    try {
      const response = await axios.post(`${this.baseUrl}/api/v1/nautilus/docker/execute`, {
        command: 'catalog',
        args: command
      })
      return response.data
    } catch (error) {
      console.error('Failed to execute Nautilus catalog command:', error)
      throw new Error('Unable to execute Nautilus catalog command')
    }
  }

  async getNautilusDataSummary(): Promise<{ instruments: string[]; totalRecords: number; storageSize: number }> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/v1/nautilus/docker/catalog/summary`)
      return response.data
    } catch (error) {
      console.error('Failed to get Nautilus data summary:', error)
      throw new Error('Unable to get Nautilus data summary')
    }
  }

  // Transform response methods
  private transformCatalogResponse(data: any): DataCatalog {
    return {
      ...data,
      lastUpdated: new Date(data.lastUpdated),
      instruments: data.instruments?.map(this.transformInstrumentResponse) || [],
      venues: data.venues || [],
      dataSources: data.dataSources || []
    }
  }

  private transformInstrumentResponse = (data: any): InstrumentMetadata => {
    return {
      ...data,
      dateRange: {
        start: new Date(data.dateRange.start),
        end: new Date(data.dateRange.end)
      },
      lastUpdated: new Date(data.lastUpdated),
      gaps: data.gaps?.map(this.transformGapResponse) || []
    }
  }

  private transformGapResponse = (gap: any): DataGap => {
    return {
      ...gap,
      start: new Date(gap.start),
      end: new Date(gap.end),
      detectedAt: new Date(gap.detectedAt),
      filledAt: gap.filledAt ? new Date(gap.filledAt) : undefined
    }
  }

  private transformQualityResponse = (data: any): QualityReport => {
    return {
      ...data,
      generatedAt: new Date(data.generatedAt),
      metrics: {
        ...data.metrics,
        lastUpdated: new Date(data.metrics.lastUpdated)
      },
      anomalies: data.anomalies || [],
      validationResults: data.validationResults?.map((result: any) => ({
        ...result,
        timestamp: new Date(result.timestamp)
      })) || []
    }
  }

  private buildSearchParams(filters: CatalogSearchFilters): Record<string, string> {
    const params: Record<string, string> = {}
    
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        if (typeof value === 'object' && value.start && value.end) {
          params[`${key}_start`] = value.start
          params[`${key}_end`] = value.end
        } else {
          params[key] = String(value)
        }
      }
    })
    
    return params
  }
}

// Singleton instance
export const dataCatalogService = new DataCatalogService()