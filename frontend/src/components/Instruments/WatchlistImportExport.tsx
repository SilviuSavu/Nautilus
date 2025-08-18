import React, { useState, useRef } from 'react'
import { Modal, Button, Upload, message, Space, Alert, Divider, Typography, Progress } from 'antd'
import { DownloadOutlined, UploadOutlined, ExportOutlined, ImportOutlined } from '@ant-design/icons'
import { useInstrumentStore } from './hooks/useInstrumentStore'
import { Watchlist, Instrument } from './types/instrumentTypes'

const { Text, Paragraph } = Typography

interface WatchlistImportExportProps {
  visible: boolean
  onClose: () => void
  mode: 'import' | 'export'
  watchlistId?: string
}

interface ExportFormat {
  watchlists: Array<{
    id: string
    name: string
    description?: string
    items: Array<{
      symbol: string
      venue: string
      assetClass: string
      currency: string
      notes?: string
      addedAt: string
    }>
    createdAt: string
    updatedAt: string
  }>
  exportedAt: string
  version: string
}

export const WatchlistImportExport: React.FC<WatchlistImportExportProps> = ({
  visible,
  onClose,
  mode,
  watchlistId
}) => {
  const [importing, setImporting] = useState(false)
  const [importProgress, setImportProgress] = useState(0)
  const [importedCount, setImportedCount] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { 
    watchlists, 
    getWatchlist, 
    createWatchlist, 
    addToWatchlist,
    searchInstruments 
  } = useInstrumentStore()

  const exportWatchlist = (watchlist: Watchlist) => {
    const exportData: ExportFormat = {
      watchlists: [{
        id: watchlist.id,
        name: watchlist.name,
        description: watchlist.description,
        items: watchlist.items.map(item => ({
          symbol: item.instrument.symbol,
          venue: item.instrument.venue,
          assetClass: item.instrument.assetClass,
          currency: item.instrument.currency,
          notes: item.notes,
          addedAt: item.addedAt
        })),
        createdAt: watchlist.createdAt,
        updatedAt: watchlist.updatedAt
      }],
      exportedAt: new Date().toISOString(),
      version: '1.0.0'
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    })
    
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${watchlist.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_watchlist.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
    
    message.success('Watchlist exported successfully!')
  }

  const exportAllWatchlists = () => {
    const exportData: ExportFormat = {
      watchlists: watchlists.map(watchlist => ({
        id: watchlist.id,
        name: watchlist.name,
        description: watchlist.description,
        items: watchlist.items.map(item => ({
          symbol: item.instrument.symbol,
          venue: item.instrument.venue,
          assetClass: item.instrument.assetClass,
          currency: item.instrument.currency,
          notes: item.notes,
          addedAt: item.addedAt
        })),
        createdAt: watchlist.createdAt,
        updatedAt: watchlist.updatedAt
      })),
      exportedAt: new Date().toISOString(),
      version: '1.0.0'
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    })
    
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `all_watchlists_${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
    
    message.success('All watchlists exported successfully!')
  }

  const exportAsCSV = (watchlist: Watchlist) => {
    const csvHeader = 'Symbol,Name,Venue,Asset Class,Currency,Notes,Added Date\\n'
    const csvData = watchlist.items.map(item => {
      const notes = (item.notes || '').replace(/"/g, '""') // Escape quotes
      return `"${item.instrument.symbol}","${item.instrument.name}","${item.instrument.venue}","${item.instrument.assetClass}","${item.instrument.currency}","${notes}","${new Date(item.addedAt).toLocaleDateString()}"`
    }).join('\\n')
    
    const blob = new Blob([csvHeader + csvData], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${watchlist.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_watchlist.csv`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
    
    message.success('Watchlist exported as CSV!')
  }

  const handleFileImport = async (file: File) => {
    setImporting(true)
    setImportProgress(0)
    setImportedCount(0)

    try {
      const text = await file.text()
      let importData: ExportFormat

      try {
        importData = JSON.parse(text)
      } catch (error) {
        throw new Error('Invalid JSON format')
      }

      if (!importData.watchlists || !Array.isArray(importData.watchlists)) {
        throw new Error('Invalid watchlist format')
      }

      let processedCount = 0
      const totalItems = importData.watchlists.reduce((total, wl) => total + wl.items.length, 0)

      for (const watchlistData of importData.watchlists) {
        // Create new watchlist
        const newWatchlist = createWatchlist(
          watchlistData.name + ' (Imported)',
          watchlistData.description
        )

        // Add instruments to the watchlist
        for (const itemData of watchlistData.items) {
          try {
            // Try to find the instrument by searching
            const searchResults = await searchInstruments(itemData.symbol, { maxResults: 10 })
            const matchingResult = searchResults.find(result => 
              result.instrument.symbol === itemData.symbol &&
              result.instrument.venue === itemData.venue &&
              result.instrument.assetClass === itemData.assetClass
            )

            if (matchingResult) {
              addToWatchlist(newWatchlist.id, matchingResult.instrument)
              setImportedCount(prev => prev + 1)
            } else {
              console.warn(`Instrument not found: ${itemData.symbol} (${itemData.venue})`)
            }
          } catch (error) {
            console.error(`Failed to import instrument ${itemData.symbol}:`, error)
          }

          processedCount++
          setImportProgress((processedCount / totalItems) * 100)
        }
      }

      message.success(`Import completed! ${importedCount} instruments imported.`)
    } catch (error) {
      console.error('Import failed:', error)
      message.error(`Import failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setImporting(false)
      setImportProgress(0)
    }
  }

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      handleFileImport(file)
    }
  }

  const selectedWatchlist = watchlistId ? getWatchlist(watchlistId) : null

  if (mode === 'export') {
    return (
      <Modal
        title="Export Watchlist"
        open={visible}
        onCancel={onClose}
        footer={null}
        width={600}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Alert
            message="Export Options"
            description="Choose how you want to export your watchlist data. JSON format preserves all data including notes, while CSV is compatible with spreadsheet applications."
            type="info"
            showIcon
          />

          {selectedWatchlist && (
            <>
              <div>
                <Text strong>Selected Watchlist: {selectedWatchlist.name}</Text>
                <br />
                <Text type="secondary">
                  {selectedWatchlist.items.length} instruments • Created {new Date(selectedWatchlist.createdAt).toLocaleDateString()}
                </Text>
              </div>

              <Space>
                <Button
                  type="primary"
                  icon={<DownloadOutlined />}
                  onClick={() => exportWatchlist(selectedWatchlist)}
                >
                  Export as JSON
                </Button>
                <Button
                  icon={<ExportOutlined />}
                  onClick={() => exportAsCSV(selectedWatchlist)}
                >
                  Export as CSV
                </Button>
              </Space>
            </>
          )}

          <Divider />

          <div>
            <Text strong>Export All Watchlists</Text>
            <br />
            <Text type="secondary">
              Export all {watchlists.length} watchlists in a single JSON file
            </Text>
          </div>

          <Button
            type="default"
            icon={<DownloadOutlined />}
            onClick={exportAllWatchlists}
          >
            Export All Watchlists
          </Button>
        </Space>
      </Modal>
    )
  }

  return (
    <Modal
      title="Import Watchlist"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={600}
    >
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Alert
          message="Import Watchlist Data"
          description="Import watchlist data from a JSON file exported from this application. The system will attempt to match instruments by symbol, venue, and asset class."
          type="info"
          showIcon
        />

        {importing && (
          <div>
            <Text strong>Importing watchlist...</Text>
            <Progress percent={Math.round(importProgress)} />
            <Text type="secondary">Imported {importedCount} instruments</Text>
          </div>
        )}

        {!importing && (
          <>
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <Button
                type="primary"
                icon={<UploadOutlined />}
                onClick={() => fileInputRef.current?.click()}
                size="large"
              >
                Select JSON File to Import
              </Button>
            </div>

            <Alert
              message="Import Notes"
              description={
                <div>
                  <Paragraph>
                    • Only instruments that exist in the current system will be imported
                  </Paragraph>
                  <Paragraph>
                    • Imported watchlists will have "(Imported)" added to their name
                  </Paragraph>
                  <Paragraph>
                    • Notes and custom data will be preserved when possible
                  </Paragraph>
                </div>
              }
              type="warning"
              showIcon
            />
          </>
        )}
      </Space>
    </Modal>
  )
}