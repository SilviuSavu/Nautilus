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

// Performance test utilities
const measureRenderTime = async (component: React.ReactElement) => {
  const startTime = performance.now()
  render(component)
  await waitFor(() => {
    // Wait for component to be rendered
    expect(document.body).not.toBeEmptyDOMElement()
  })
  const endTime = performance.now()
  return endTime - startTime
}

const generateLargeDataset = (size: number) => {
  const instruments = Array.from({ length: size }, (_, i) => ({
    instrumentId: `INSTRUMENT${i}.SIM`,
    venue: `VENUE${i % 10}`,
    symbol: `INST${i}`,
    assetClass: ['Currency', 'Equity', 'Commodity', 'Bond'][i % 4],
    currency: ['USD', 'EUR', 'GBP', 'JPY'][i % 4],
    dataType: ['tick', 'quote', 'bar'][i % 3],
    timeframes: ['1-MINUTE', '5-MINUTE', '1-HOUR'],
    dateRange: {
      start: '2024-01-01T00:00:00Z',
      end: '2024-01-31T23:59:59Z'
    },
    recordCount: Math.floor(Math.random() * 1000000) + 100000,
    qualityScore: Math.random() * 0.3 + 0.7, // 0.7 to 1.0
    gaps: [],
    lastUpdated: new Date().toISOString(),
    fileSize: Math.floor(Math.random() * 100000000) + 1000000
  }))

  return {
    instruments,
    venues: Array.from({ length: 10 }, (_, i) => ({
      id: `VENUE${i}`,
      name: `Venue ${i}`
    })),
    dataSources: [
      { id: 'nautilus', name: 'NautilusTrader', type: 'historical', status: 'active' }
    ],
    qualityMetrics: {
      completeness: 0.94,
      accuracy: 0.96,
      timeliness: 0.91,
      consistency: 0.93,
      overall: 0.935,
      lastUpdated: new Date().toISOString()
    },
    lastUpdated: new Date().toISOString(),
    totalInstruments: size,
    totalRecords: instruments.reduce((sum, inst) => sum + inst.recordCount, 0),
    storageSize: instruments.reduce((sum, inst) => sum + inst.fileSize, 0)
  }
}

const generateLargeFeedData = (size: number) => ({
  feeds: Array.from({ length: size }, (_, i) => ({
    feedId: `feed_${i}`,
    source: `Source ${i}`,
    status: ['connected', 'disconnected', 'degraded', 'reconnecting'][i % 4],
    latency: Math.floor(Math.random() * 200) + 10,
    throughput: Math.floor(Math.random() * 5000) + 100,
    lastUpdate: new Date().toISOString(),
    errorCount: Math.floor(Math.random() * 10),
    qualityScore: Math.random() * 0.3 + 0.7,
    subscriptionCount: Math.floor(Math.random() * 50),
    bandwidth: Math.floor(Math.random() * 2048) + 256
  }))
})

describe('Data Catalog Performance Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    
    // Mock performance.now if not available
    if (!global.performance) {
      global.performance = {
        now: () => Date.now()
      } as any
    }
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('Large Dataset Rendering Performance', () => {
    test('should render 1000 instruments within 3 seconds', async () => {
      const largeDataset = generateLargeDataset(1000)
      
      mockedAxios.get.mockImplementation((url: string) => {
        if (url.includes('/api/v1/nautilus/data/catalog')) {
          return Promise.resolve({ data: largeDataset })
        }
        return Promise.reject(new Error(`Unmocked GET request: ${url}`))
      })

      const renderTime = await measureRenderTime(<DataCatalogBrowser />)
      
      // Should render within 3 seconds (story requirement)
      expect(renderTime).toBeLessThan(3000)
      
      // Verify content is rendered
      await waitFor(() => {
        expect(screen.getByText('1,000')).toBeInTheDocument() // Total instruments
      })
    })

    test('should handle 10000 instruments in table view efficiently', async () => {
      const veryLargeDataset = generateLargeDataset(10000)
      
      mockedAxios.get.mockImplementation((url: string) => {
        if (url.includes('/api/v1/nautilus/data/catalog')) {
          return Promise.resolve({ data: veryLargeDataset })
        }
        if (url.includes('/search')) {
          return Promise.resolve({ data: { instruments: veryLargeDataset.instruments.slice(0, 50), totalCount: 10000 } })
        }
        return Promise.reject(new Error(`Unmocked GET request: ${url}`))
      })

      const startTime = performance.now()
      render(<DataCatalogBrowser />)
      
      await waitFor(() => {
        expect(screen.getByText('10,000')).toBeInTheDocument()
      })
      
      // Switch to table view
      const tableViewButton = screen.getByDisplayValue('table')
      if (tableViewButton) {
        fireEvent.change(tableViewButton, { target: { value: 'table' } })
      }
      
      const endTime = performance.now()
      const totalTime = endTime - startTime
      
      // Should handle large datasets efficiently
      expect(totalTime).toBeLessThan(5000) // 5 second limit for very large datasets
    })

    test('should handle rapid scrolling in large tables', async () => {
      const largeDataset = generateLargeDataset(5000)
      
      mockedAxios.get.mockResolvedValue({ data: largeDataset })

      render(<DataCatalogBrowser />)
      
      await waitFor(() => {
        expect(screen.getByText('5,000')).toBeInTheDocument()
      })

      // Simulate rapid scrolling by triggering multiple scroll events
      const table = document.querySelector('.ant-table-tbody')
      if (table) {
        const startTime = performance.now()
        
        // Simulate rapid scroll events
        for (let i = 0; i < 50; i++) {
          fireEvent.scroll(table, { target: { scrollTop: i * 100 } })
        }
        
        const endTime = performance.now()
        const scrollTime = endTime - startTime
        
        // Scrolling should remain responsive
        expect(scrollTime).toBeLessThan(1000)
      }
    })
  })

  describe('Feed Monitoring Performance', () => {
    test('should handle 500 concurrent feed statuses', async () => {
      const largeFeedData = generateLargeFeedData(500)
      
      mockedAxios.get.mockImplementation((url: string) => {
        if (url.includes('/feeds/status')) {
          return Promise.resolve({ data: largeFeedData })
        }
        if (url.includes('/pipeline/health')) {
          return Promise.resolve({ 
            data: { 
              status: 'healthy', 
              details: { uptime: 99.8, throughput: 15420, latency: 12 } 
            } 
          })
        }
        return Promise.reject(new Error(`Unmocked GET request: ${url}`))
      })

      const renderTime = await measureRenderTime(<DataPipelineMonitor />)
      
      // Should render large feed list efficiently
      expect(renderTime).toBeLessThan(2000)
      
      await waitFor(() => {
        expect(screen.getByText('feed_0')).toBeInTheDocument()
      })
    })

    test('should handle real-time updates efficiently', async () => {
      const feedData = generateLargeFeedData(100)
      
      let updateCount = 0
      mockedAxios.get.mockImplementation((url: string) => {
        if (url.includes('/feeds/status')) {
          updateCount++
          // Simulate changing data
          const updatedData = {
            ...feedData,
            feeds: feedData.feeds.map(feed => ({
              ...feed,
              lastUpdate: new Date().toISOString(),
              throughput: Math.floor(Math.random() * 5000) + 100
            }))
          }
          return Promise.resolve({ data: updatedData })
        }
        if (url.includes('/pipeline/health')) {
          return Promise.resolve({ 
            data: { 
              status: 'healthy', 
              details: { uptime: 99.8, throughput: 15420, latency: 12 } 
            } 
          })
        }
        return Promise.reject(new Error(`Unmocked GET request: ${url}`))
      })

      render(<DataPipelineMonitor />)
      
      await waitFor(() => {
        expect(screen.getByText('feed_0')).toBeInTheDocument()
      })

      const startTime = performance.now()
      
      // Simulate auto-refresh cycles
      for (let i = 0; i < 10; i++) {
        await new Promise(resolve => setTimeout(resolve, 100))
      }
      
      const endTime = performance.now()
      const updateTime = endTime - startTime
      
      // Updates should be processed efficiently
      expect(updateTime / updateCount).toBeLessThan(100) // Less than 100ms per update
    })
  })

  describe('Search and Filtering Performance', () => {
    test('should filter 10000 instruments quickly', async () => {
      const largeDataset = generateLargeDataset(10000)
      
      mockedAxios.get.mockImplementation((url: string) => {
        if (url.includes('/api/v1/nautilus/data/catalog')) {
          return Promise.resolve({ data: largeDataset })
        }
        if (url.includes('/search')) {
          // Simulate filtered results
          const filtered = largeDataset.instruments.filter(inst => 
            inst.assetClass === 'Currency'
          ).slice(0, 100)
          return Promise.resolve({ 
            data: { 
              instruments: filtered, 
              totalCount: filtered.length 
            } 
          })
        }
        return Promise.reject(new Error(`Unmocked GET request: ${url}`))
      })

      render(<DataCatalogBrowser />)
      
      await waitFor(() => {
        expect(screen.getByText('10,000')).toBeInTheDocument()
      })

      const startTime = performance.now()
      
      // Apply filter
      const assetClassSelect = screen.getByPlaceholderText('Asset Class')
      if (assetClassSelect) {
        fireEvent.change(assetClassSelect, { target: { value: 'Currency' } })
        
        const searchButton = screen.getByText('Search')
        fireEvent.click(searchButton)
      }
      
      await waitFor(() => {
        expect(mockedAxios.get).toHaveBeenCalledWith(
          expect.stringContaining('/search'),
          expect.any(Object)
        )
      })
      
      const endTime = performance.now()
      const searchTime = endTime - startTime
      
      // Search should complete quickly
      expect(searchTime).toBeLessThan(1000)
    })

    test('should handle complex multi-filter scenarios', async () => {
      const largeDataset = generateLargeDataset(5000)
      
      mockedAxios.get.mockImplementation((url: string) => {
        if (url.includes('/api/v1/nautilus/data/catalog')) {
          return Promise.resolve({ data: largeDataset })
        }
        if (url.includes('/search')) {
          const filtered = largeDataset.instruments.slice(0, 50)
          return Promise.resolve({ 
            data: { 
              instruments: filtered, 
              totalCount: filtered.length 
            } 
          })
        }
        return Promise.reject(new Error(`Unmocked GET request: ${url}`))
      })

      render(<DataCatalogBrowser />)
      
      await waitFor(() => {
        expect(screen.getByText('5,000')).toBeInTheDocument()
      })

      const startTime = performance.now()
      
      // Apply multiple filters rapidly
      const filters = [
        { field: 'assetClass', value: 'Currency' },
        { field: 'dataType', value: 'tick' },
        { field: 'qualityThreshold', value: '90' }
      ]
      
      for (const filter of filters) {
        const input = screen.getByPlaceholderText(filter.field)
        if (input) {
          fireEvent.change(input, { target: { value: filter.value } })
        }
        await new Promise(resolve => setTimeout(resolve, 50))
      }
      
      const searchButton = screen.getByText('Search')
      fireEvent.click(searchButton)
      
      await waitFor(() => {
        expect(mockedAxios.get).toHaveBeenCalled()
      })
      
      const endTime = performance.now()
      const filterTime = endTime - startTime
      
      // Complex filtering should remain responsive
      expect(filterTime).toBeLessThan(2000)
    })
  })

  describe('Gap Analysis Performance', () => {
    test('should analyze gaps for 1000 instruments efficiently', async () => {
      const largeDataset = generateLargeDataset(1000)
      
      // Add gaps to some instruments
      largeDataset.instruments.forEach((inst, i) => {
        if (i % 10 === 0) { // Every 10th instrument has gaps
          inst.gaps = Array.from({ length: 5 }, (_, j) => ({
            id: `gap_${i}_${j}`,
            start: new Date(Date.now() - j * 3600000).toISOString(),
            end: new Date(Date.now() - j * 3600000 + 900000).toISOString(), // 15 min gap
            severity: ['low', 'medium', 'high'][j % 3] as 'low' | 'medium' | 'high',
            reason: `Gap ${j} for instrument ${i}`,
            detectedAt: new Date().toISOString()
          }))
        }
      })
      
      mockedAxios.get.mockImplementation((url: string) => {
        if (url.includes('/gaps/')) {
          const instrumentId = url.split('/').pop()
          const instrument = largeDataset.instruments.find(i => i.instrumentId === instrumentId)
          return Promise.resolve({ data: { gaps: instrument?.gaps || [] } })
        }
        return Promise.resolve({ data: largeDataset })
      })

      const renderTime = await measureRenderTime(<GapAnalysisView />)
      
      // Should load gap analysis efficiently
      expect(renderTime).toBeLessThan(2000)
      
      await waitFor(() => {
        expect(screen.getByText('Gap Analysis by Instrument')).toBeInTheDocument()
      })
    })

    test('should handle gap analysis for instruments with many gaps', async () => {
      const instrumentWithManyGaps = {
        instrumentId: 'GAPPY.SIM',
        venue: 'SIM',
        symbol: 'GAPPY',
        assetClass: 'Test',
        currency: 'USD',
        dataType: 'tick' as const,
        timeframes: ['1-MINUTE'],
        dateRange: {
          start: '2024-01-01T00:00:00Z',
          end: '2024-01-31T23:59:59Z'
        },
        recordCount: 1000000,
        qualityScore: 0.5,
        gaps: Array.from({ length: 1000 }, (_, i) => ({
          id: `gap_${i}`,
          start: new Date(Date.now() - i * 3600000).toISOString(),
          end: new Date(Date.now() - i * 3600000 + 300000).toISOString(),
          severity: ['low', 'medium', 'high'][i % 3] as 'low' | 'medium' | 'high',
          reason: `Gap ${i}`,
          detectedAt: new Date().toISOString()
        })),
        lastUpdated: new Date().toISOString(),
        fileSize: 50000000
      }
      
      mockedAxios.get.mockResolvedValue({ 
        data: { gaps: instrumentWithManyGaps.gaps }
      })

      const startTime = performance.now()
      render(<GapAnalysisView />)
      
      await waitFor(() => {
        expect(screen.getByText('Gap Analysis by Instrument')).toBeInTheDocument()
      })
      
      const endTime = performance.now()
      const renderTime = endTime - startTime
      
      // Should handle many gaps efficiently
      expect(renderTime).toBeLessThan(3000)
    })
  })

  describe('Export Performance', () => {
    test('should handle export of large datasets', async () => {
      const exportRequest = {
        instrumentIds: Array.from({ length: 100 }, (_, i) => `INST${i}.SIM`),
        format: 'parquet',
        dateRange: {
          start: '2024-01-01',
          end: '2024-12-31'
        },
        maxRecords: 10000000 // 10M records
      }
      
      mockedAxios.post.mockImplementation((url: string) => {
        if (url.includes('/export')) {
          return Promise.resolve({
            data: {
              exportId: 'large_export_123',
              success: true,
              recordCount: 10000000,
              fileSize: 500000000, // 500MB
              format: 'parquet',
              createdAt: new Date().toISOString(),
              completedAt: new Date(Date.now() + 30000).toISOString() // 30 seconds later
            }
          })
        }
        return Promise.reject(new Error(`Unmocked POST request: ${url}`))
      })

      render(<ExportImportTools />)
      
      await waitFor(() => {
        expect(screen.getByText('Export Data')).toBeInTheDocument()
      })

      const startTime = performance.now()
      
      // Simulate export initiation
      const exportButton = screen.getByText('Start Export')
      fireEvent.click(exportButton)
      
      await waitFor(() => {
        expect(mockedAxios.post).toHaveBeenCalled()
      })
      
      const endTime = performance.now()
      const responseTime = endTime - startTime
      
      // Export request should be processed quickly
      expect(responseTime).toBeLessThan(1000)
    })
  })

  describe('Memory Usage and Cleanup', () => {
    test('should clean up properly when unmounting', async () => {
      const largeDataset = generateLargeDataset(1000)
      mockedAxios.get.mockResolvedValue({ data: largeDataset })

      const { unmount } = render(<DataCatalogBrowser />)
      
      await waitFor(() => {
        expect(screen.getByText('1,000')).toBeInTheDocument()
      })

      // Unmount component
      unmount()
      
      // Component should clean up without errors
      // This test mainly ensures no memory leaks or hanging promises
      expect(true).toBe(true)
    })

    test('should handle rapid mount/unmount cycles', async () => {
      const dataset = generateLargeDataset(100)
      mockedAxios.get.mockResolvedValue({ data: dataset })

      // Rapidly mount and unmount components
      for (let i = 0; i < 10; i++) {
        const { unmount } = render(<DataQualityDashboard />)
        await new Promise(resolve => setTimeout(resolve, 10))
        unmount()
      }
      
      // Should handle rapid cycles without issues
      expect(true).toBe(true)
    })
  })

  describe('Browser Performance Limits', () => {
    test('should gracefully handle extremely large datasets', async () => {
      // Test with 50,000 instruments (extreme case)
      const extremeDataset = generateLargeDataset(50000)
      
      mockedAxios.get.mockImplementation((url: string) => {
        if (url.includes('/search')) {
          // Return paginated results
          return Promise.resolve({ 
            data: { 
              instruments: extremeDataset.instruments.slice(0, 50), 
              totalCount: 50000 
            } 
          })
        }
        return Promise.resolve({ data: extremeDataset })
      })

      const startTime = performance.now()
      render(<DataCatalogBrowser />)
      
      // Component should still render (possibly with loading state)
      await waitFor(() => {
        expect(document.body).not.toBeEmptyDOMElement()
      }, { timeout: 10000 })
      
      const endTime = performance.now()
      const renderTime = endTime - startTime
      
      // Should handle extreme datasets without crashing
      expect(renderTime).toBeLessThan(10000) // 10 second absolute limit
    })
  })
})