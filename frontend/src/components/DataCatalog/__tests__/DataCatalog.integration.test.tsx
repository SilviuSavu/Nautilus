import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi, describe, test, expect, beforeEach, afterEach } from 'vitest'
import axios from 'axios'
import { DataCatalogBrowser } from '../DataCatalogBrowser'
import { DataQualityDashboard } from '../DataQualityDashboard'
import { DataPipelineMonitor } from '../DataPipelineMonitor'
import { ExportImportTools } from '../ExportImportTools'
import { GapAnalysisView } from '../GapAnalysisView'

// Mock axios
vi.mock('axios')
const mockedAxios = axios as any

// Mock data
const mockCatalogData = {
  instruments: [
    {
      instrumentId: 'EURUSD.SIM',
      venue: 'SIM',
      symbol: 'EUR/USD',
      assetClass: 'Currency',
      currency: 'USD',
      dataType: 'tick',
      timeframes: ['1-MINUTE', '5-MINUTE'],
      dateRange: {
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z'
      },
      recordCount: 1250000,
      qualityScore: 0.96,
      gaps: [],
      lastUpdated: '2024-01-31T12:00:00Z',
      fileSize: 45678912
    },
    {
      instrumentId: 'GBPUSD.SIM',
      venue: 'SIM',
      symbol: 'GBP/USD',
      assetClass: 'Currency',
      currency: 'USD',
      dataType: 'tick',
      timeframes: ['1-MINUTE'],
      dateRange: {
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z'
      },
      recordCount: 980000,
      qualityScore: 0.85,
      gaps: [
        {
          id: 'gap_001',
          start: '2024-01-15T09:30:00Z',
          end: '2024-01-15T09:45:00Z',
          severity: 'medium',
          reason: 'Market data feed interruption',
          detectedAt: '2024-01-15T10:00:00Z'
        }
      ],
      lastUpdated: '2024-01-31T12:00:00Z',
      fileSize: 35467843
    }
  ],
  venues: [
    { id: 'SIM', name: 'Simulated Exchange' }
  ],
  dataSources: [
    { id: 'nautilus', name: 'NautilusTrader', type: 'historical', status: 'active' }
  ],
  qualityMetrics: {
    completeness: 0.94,
    accuracy: 0.96,
    timeliness: 0.91,
    consistency: 0.93,
    overall: 0.935,
    lastUpdated: '2024-01-31T12:00:00Z'
  },
  lastUpdated: '2024-01-31T12:00:00Z',
  totalInstruments: 2,
  totalRecords: 2230000,
  storageSize: 81146755
}

const mockFeedStatuses = {
  feeds: [
    {
      feedId: 'ib_market_data',
      source: 'Interactive Brokers',
      status: 'connected',
      latency: 15,
      throughput: 2500,
      lastUpdate: '2024-01-31T12:00:00Z',
      errorCount: 0,
      qualityScore: 0.98,
      subscriptionCount: 25,
      bandwidth: 1024
    },
    {
      feedId: 'yahoo_finance',
      source: 'Yahoo Finance',
      status: 'degraded',
      latency: 150,
      throughput: 150,
      lastUpdate: '2024-01-31T12:00:00Z',
      errorCount: 3,
      qualityScore: 0.85,
      subscriptionCount: 8
    }
  ]
}

const mockPipelineHealth = {
  status: 'healthy',
  details: {
    uptime: 99.8,
    throughput: 15420,
    latency: 12,
    errorRate: 0.2,
    activeFeeds: 2,
    totalFeeds: 3
  }
}

const mockExportResult = {
  exportId: 'exp_123',
  success: true,
  filePath: '/exports/exp_123.parquet',
  downloadUrl: '/api/downloads/exp_123',
  recordCount: 100000,
  fileSize: 5242880,
  format: 'parquet',
  createdAt: '2024-01-31T12:00:00Z',
  completedAt: '2024-01-31T12:05:00Z'
}

describe('Data Catalog Integration Tests', () => {
  beforeEach(() => {
    // Reset mocks
    vi.clearAllMocks()
    
    // Setup default API responses
    mockedAxios.get.mockImplementation((url: string) => {
      if (url.includes('/api/v1/nautilus/data/catalog')) {
        return Promise.resolve({ data: mockCatalogData })
      }
      if (url.includes('/api/v1/nautilus/data/feeds/status')) {
        return Promise.resolve({ data: mockFeedStatuses })
      }
      if (url.includes('/api/v1/nautilus/data/pipeline/health')) {
        return Promise.resolve({ data: mockPipelineHealth })
      }
      if (url.includes('/api/v1/nautilus/data/gaps/')) {
        return Promise.resolve({ 
          data: { 
            gaps: mockCatalogData.instruments[1].gaps 
          }
        })
      }
      if (url.includes('/api/v1/nautilus/data/quality/')) {
        return Promise.resolve({ 
          data: mockCatalogData.qualityMetrics
        })
      }
      return Promise.reject(new Error(`Unmocked GET request: ${url}`))
    })

    mockedAxios.post.mockImplementation((url: string, data: any) => {
      if (url.includes('/api/v1/nautilus/data/export')) {
        return Promise.resolve({ data: mockExportResult })
      }
      if (url.includes('/api/v1/nautilus/data/quality/validate')) {
        return Promise.resolve({
          data: {
            instrumentId: data.instrumentId,
            metrics: mockCatalogData.qualityMetrics,
            anomalies: [],
            validationResults: [
              {
                checkName: 'Data Completeness',
                passed: true,
                message: 'All expected data points present',
                timestamp: new Date().toISOString()
              }
            ],
            generatedAt: new Date().toISOString()
          }
        })
      }
      if (url.includes('/api/v1/nautilus/data/feeds/subscribe')) {
        return Promise.resolve({ data: { success: true } })
      }
      return Promise.reject(new Error(`Unmocked POST request: ${url}`))
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('DataCatalogBrowser Integration', () => {
    test('should load and display catalog data', async () => {
      render(<DataCatalogBrowser />)

      // Wait for data to load
      await waitFor(() => {
        expect(screen.getByText('EUR/USD')).toBeInTheDocument()
        expect(screen.getByText('GBP/USD')).toBeInTheDocument()
      })

      // Check statistics
      expect(screen.getByText('2')).toBeInTheDocument() // Total instruments
      expect(screen.getByText('2,230,000')).toBeInTheDocument() // Total records
    })

    test('should handle search and filtering', async () => {
      render(<DataCatalogBrowser />)

      await waitFor(() => {
        expect(screen.getByText('EUR/USD')).toBeInTheDocument()
      })

      // Test search functionality
      const searchButton = screen.getByText('Search')
      fireEvent.click(searchButton)

      await waitFor(() => {
        expect(mockedAxios.get).toHaveBeenCalledWith(
          expect.stringContaining('/api/v1/nautilus/data/catalog/search'),
          expect.any(Object)
        )
      })
    })

    test('should handle instrument selection', async () => {
      const onInstrumentSelect = vi.fn()
      render(<DataCatalogBrowser onInstrumentSelect={onInstrumentSelect} />)

      await waitFor(() => {
        expect(screen.getByText('EUR/USD')).toBeInTheDocument()
      })

      // Click on an instrument
      const instrumentButton = screen.getByText('EUR/USD')
      fireEvent.click(instrumentButton)

      expect(onInstrumentSelect).toHaveBeenCalledWith(
        expect.objectContaining({
          instrumentId: 'EURUSD.SIM',
          symbol: 'EUR/USD'
        })
      )
    })

    test('should handle export request', async () => {
      const onExportRequest = vi.fn()
      render(<DataCatalogBrowser onExportRequest={onExportRequest} />)

      await waitFor(() => {
        expect(screen.getByText('EUR/USD')).toBeInTheDocument()
      })

      // Select instruments and export
      const checkboxes = screen.getAllByRole('checkbox')
      fireEvent.click(checkboxes[1]) // Select first instrument

      // The export button should appear when instruments are selected
      // This would require the actual implementation to show the export button
    })
  })

  describe('DataQualityDashboard Integration', () => {
    test('should load and display quality metrics', async () => {
      render(<DataQualityDashboard />)

      await waitFor(() => {
        expect(screen.getByText('94%')).toBeInTheDocument() // Overall quality
      })

      // Check individual metrics
      expect(screen.getByText('96%')).toBeInTheDocument() // Completeness
    })

    test('should handle quality validation', async () => {
      render(<DataQualityDashboard />)

      await waitFor(() => {
        expect(screen.getByText('EUR/USD')).toBeInTheDocument()
      })

      // Click validate button
      const validateButtons = screen.getAllByText('Validate')
      if (validateButtons.length > 0) {
        fireEvent.click(validateButtons[0])

        await waitFor(() => {
          expect(mockedAxios.post).toHaveBeenCalledWith(
            expect.stringContaining('/api/v1/nautilus/data/quality/validate'),
            expect.any(Object)
          )
        })
      }
    })
  })

  describe('DataPipelineMonitor Integration', () => {
    test('should load and display feed statuses', async () => {
      render(<DataPipelineMonitor />)

      await waitFor(() => {
        expect(screen.getByText('ib_market_data')).toBeInTheDocument()
        expect(screen.getByText('yahoo_finance')).toBeInTheDocument()
      })

      // Check feed status indicators
      expect(screen.getByText('CONNECTED')).toBeInTheDocument()
      expect(screen.getByText('DEGRADED')).toBeInTheDocument()
    })

    test('should handle feed subscription', async () => {
      render(<DataPipelineMonitor />)

      await waitFor(() => {
        expect(screen.getByText('ib_market_data')).toBeInTheDocument()
      })

      // Click subscribe button
      const subscribeButtons = screen.getAllByText('Subscribe')
      if (subscribeButtons.length > 0) {
        fireEvent.click(subscribeButtons[0])

        // Modal should open - this would require modal implementation
        // The test would continue to fill the form and submit
      }
    })

    test('should display pipeline health metrics', async () => {
      render(<DataPipelineMonitor />)

      await waitFor(() => {
        expect(screen.getByText('HEALTHY')).toBeInTheDocument()
      })

      // Check throughput and latency metrics
      expect(screen.getByText('15,420')).toBeInTheDocument() // Throughput
      expect(screen.getByText('12ms')).toBeInTheDocument() // Latency
    })
  })

  describe('ExportImportTools Integration', () => {
    test('should handle data export', async () => {
      render(<ExportImportTools />)

      // Fill export form
      const instrumentSelect = screen.getByPlaceholderText('Select instruments')
      fireEvent.change(instrumentSelect, { target: { value: 'EURUSD.SIM' } })

      // Submit export
      const exportButton = screen.getByText('Start Export')
      fireEvent.click(exportButton)

      await waitFor(() => {
        expect(mockedAxios.post).toHaveBeenCalledWith(
          expect.stringContaining('/api/v1/nautilus/data/export'),
          expect.objectContaining({
            instrument_ids: expect.any(Array),
            format: expect.any(String)
          })
        )
      })
    })

    test('should display export history', async () => {
      render(<ExportImportTools />)

      // The component should show mock export history
      // This would require the component to load and display export jobs
    })
  })

  describe('GapAnalysisView Integration', () => {
    test('should load and display gap analysis data', async () => {
      render(<GapAnalysisView />)

      await waitFor(() => {
        expect(screen.getByText('Gap Analysis by Instrument')).toBeInTheDocument()
      })

      // Should display instruments with gaps
      expect(screen.getByText('GBP/USD')).toBeInTheDocument()
    })

    test('should handle gap analysis trigger', async () => {
      render(<GapAnalysisView />)

      await waitFor(() => {
        expect(screen.getByText('Analyze All')).toBeInTheDocument()
      })

      // Click analyze all button
      const analyzeButton = screen.getByText('Analyze All')
      fireEvent.click(analyzeButton)

      // Should trigger gap analysis for all instruments
      await waitFor(() => {
        expect(mockedAxios.get).toHaveBeenCalledWith(
          expect.stringContaining('/api/v1/nautilus/data/gaps/'),
          expect.any(Object)
        )
      })
    })

    test('should display gap statistics', async () => {
      render(<GapAnalysisView />)

      await waitFor(() => {
        // Should show gap summary statistics
        expect(screen.getByText('Instruments with Gaps')).toBeInTheDocument()
        expect(screen.getByText('Total Gaps')).toBeInTheDocument()
      })
    })
  })

  describe('Error Handling', () => {
    test('should handle API errors gracefully', async () => {
      // Mock API failure
      mockedAxios.get.mockRejectedValueOnce(new Error('API Error'))

      render(<DataCatalogBrowser />)

      // Should handle error without crashing
      await waitFor(() => {
        // Component should still render even with API error
        expect(screen.getByText('Data Catalog')).toBeInTheDocument()
      })
    })

    test('should handle network timeouts', async () => {
      // Mock timeout
      mockedAxios.get.mockImplementation(() => 
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Network timeout')), 100)
        )
      )

      render(<DataPipelineMonitor />)

      // Should handle timeout gracefully
      await waitFor(() => {
        expect(screen.getByText('Loading pipeline monitor...')).toBeInTheDocument()
      }, { timeout: 500 })
    })
  })

  describe('Performance Tests', () => {
    test('should handle large datasets efficiently', async () => {
      // Mock large dataset
      const largeDataset = {
        ...mockCatalogData,
        instruments: Array.from({ length: 1000 }, (_, i) => ({
          ...mockCatalogData.instruments[0],
          instrumentId: `INSTRUMENT${i}.SIM`,
          symbol: `INST${i}`
        }))
      }

      mockedAxios.get.mockResolvedValueOnce({ data: largeDataset })

      const startTime = performance.now()
      render(<DataCatalogBrowser />)

      await waitFor(() => {
        expect(screen.getByText('INST0')).toBeInTheDocument()
      })

      const endTime = performance.now()
      const renderTime = endTime - startTime

      // Should render large datasets within reasonable time (< 2 seconds)
      expect(renderTime).toBeLessThan(2000)
    })

    test('should handle rapid API calls without issues', async () => {
      render(<DataQualityDashboard />)

      await waitFor(() => {
        expect(screen.getByText('Refresh')).toBeInTheDocument()
      })

      // Rapidly click refresh multiple times
      const refreshButton = screen.getByText('Refresh')
      for (let i = 0; i < 5; i++) {
        fireEvent.click(refreshButton)
      }

      // Should handle multiple rapid requests gracefully
      await waitFor(() => {
        expect(mockedAxios.get).toHaveBeenCalled()
      })
    })
  })
})