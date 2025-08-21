/**
 * Multi-Chart View Component
 * Renders multiple charts in a responsive grid layout with synchronization
 */

import React, { useState, useRef, useEffect, useMemo } from 'react'
import { Card, Button, Space, Tooltip, Modal, Select, message } from 'antd'
import { PlusOutlined, SettingOutlined, DeleteOutlined, DragOutlined, LinkOutlined } from '@ant-design/icons'
import { Responsive, WidthProvider, Layout as GridLayout, Layouts } from 'react-grid-layout'
import { ChartLayout, ChartConfig, Instrument, ChartType } from '../../types/charting'
import { OHLCVData } from '../Chart/types/chartTypes'
import AdvancedChartContainer from '../AdvancedChart/ChartContainer'
import ChartTypeSelector from '../AdvancedChart/ChartTypeSelector'
import { chartLayoutService, SynchronizationGroup } from '../../services/chartLayoutService'

import 'react-grid-layout/css/styles.css'
import 'react-resizable/css/styles.css'

const { Option } = Select
const ResponsiveGridLayout = WidthProvider(Responsive)

interface MultiChartViewProps {
  layout: ChartLayout
  instruments: Instrument[]
  chartData: Map<string, OHLCVData[]> // keyed by instrument symbol
  onLayoutChange?: (layout: ChartLayout) => void
  onChartAdd?: (chartConfig: ChartConfig) => void
  onChartRemove?: (chartId: string) => void
  theme?: 'light' | 'dark'
}

interface ChartSyncState {
  crosshairPosition?: { x: number; y: number; time: string }
  zoomRange?: { from: string; to: string }
  timeRange?: { from: string; to: string }
}

export const MultiChartView: React.FC<MultiChartViewProps> = ({
  layout,
  instruments,
  chartData,
  onLayoutChange,
  onChartAdd,
  onChartRemove,
  theme = 'light'
}) => {
  const [gridLayouts, setGridLayouts] = useState<Layouts>({})
  const [synchronizationGroups, setSynchronizationGroups] = useState<SynchronizationGroup[]>([])
  const [syncState, setSyncState] = useState<ChartSyncState>({})
  const [isAddChartModalVisible, setIsAddChartModalVisible] = useState(false)
  const [selectedInstrument, setSelectedInstrument] = useState<string>('')
  const [selectedChartType, setSelectedChartType] = useState<ChartType>('candlestick')
  const [isDragging, setIsDragging] = useState(false)
  const [selectedCharts, setSelectedCharts] = useState<string[]>([])

  const chartRefs = useRef<Map<string, any>>(new Map())

  useEffect(() => {
    // Initialize grid layout from chart layout
    const gridLayout = layout.charts.map(chart => {
      const position = layout.layout.chartPositions.find(pos => pos.chartId === chart.id)
      return {
        i: chart.id,
        x: position?.column || 0,
        y: position?.row || 0,
        w: position?.columnSpan || 1,
        h: position?.rowSpan || 1,
        minW: 1,
        minH: 1
      }
    })

    setGridLayouts({
      lg: gridLayout,
      md: gridLayout,
      sm: gridLayout,
      xs: gridLayout,
      xxs: gridLayout
    })
  }, [layout])

  useEffect(() => {
    // Load synchronization groups
    setSynchronizationGroups(chartLayoutService.getSynchronizationGroups())
    
    // Set up sync event handlers
    chartLayoutService.setEventHandlers({
      onSynchronizationChange: setSynchronizationGroups
    })
  }, [])

  const handleGridLayoutChange = (newLayout: GridLayout[], allLayouts: Layouts) => {
    if (isDragging) return // Don't update during drag

    // Update chart positions in the layout
    const updatedPositions = newLayout.map(item => ({
      chartId: item.i,
      row: item.y,
      column: item.x,
      rowSpan: item.h,
      columnSpan: item.w
    }))

    const updatedLayout: ChartLayout = {
      ...layout,
      layout: {
        ...layout.layout,
        chartPositions: updatedPositions
      }
    }

    chartLayoutService.updateLayout(layout.id, updatedLayout)
    onLayoutChange?.(updatedLayout)
    setGridLayouts(allLayouts)
  }

  const handleAddChart = () => {
    if (!selectedInstrument) {
      message.error('Please select an instrument')
      return
    }

    const instrument = instruments.find(inst => inst.symbol === selectedInstrument)
    if (!instrument) {
      message.error('Invalid instrument selected')
      return
    }

    const chartConfig: Omit<ChartConfig, 'id'> = {
      instrument,
      chartType: selectedChartType,
      timeframe: '1d',
      indicators: [],
      drawings: [],
      theme: 'default'
    }

    const chartId = chartLayoutService.addChartToLayout(layout.id, chartConfig)
    if (chartId) {
      onChartAdd?.({
        id: chartId,
        ...chartConfig
      })
      setIsAddChartModalVisible(false)
      setSelectedInstrument('')
      setSelectedChartType('candlestick')
      message.success('Chart added successfully')
    }
  }

  const handleRemoveChart = (chartId: string) => {
    Modal.confirm({
      title: 'Remove Chart',
      content: 'Are you sure you want to remove this chart?',
      onOk: () => {
        chartLayoutService.removeChartFromLayout(layout.id, chartId)
        onChartRemove?.(chartId)
        message.success('Chart removed')
      }
    })
  }

  const handleCreateSyncGroup = () => {
    if (selectedCharts.length < 2) {
      message.error('Select at least 2 charts to create a sync group')
      return
    }

    const syncSettings = {
      crosshair: layout.synchronization.crosshair,
      zoom: layout.synchronization.zoom,
      timeRange: layout.synchronization.timeRange,
      instrument: false
    }

    chartLayoutService.createSynchronizationGroup(selectedCharts, syncSettings)
    setSelectedCharts([])
    message.success('Synchronization group created')
  }

  const handleChartSelect = (chartId: string, selected: boolean) => {
    setSelectedCharts(prev => {
      if (selected) {
        return [...prev, chartId]
      } else {
        return prev.filter(id => id !== chartId)
      }
    })
  }

  const handleCrosshairMove = (chartId: string, position: { x: number; y: number; time: string }) => {
    const syncGroup = chartLayoutService.getSyncGroupForChart(chartId)
    if (syncGroup?.syncSettings.crosshair) {
      setSyncState(prev => ({ ...prev, crosshairPosition: position }))
      
      // Notify other charts in sync group
      syncGroup.chartIds.forEach(id => {
        if (id !== chartId) {
          const chartRef = chartRefs.current.get(id)
          if (chartRef && chartRef.setCrosshairPosition) {
            chartRef.setCrosshairPosition(position)
          }
        }
      })
    }
  }

  const handleZoomChange = (chartId: string, range: { from: string; to: string }) => {
    const syncGroup = chartLayoutService.getSyncGroupForChart(chartId)
    if (syncGroup?.syncSettings.zoom) {
      setSyncState(prev => ({ ...prev, zoomRange: range }))
      
      // Notify other charts in sync group
      syncGroup.chartIds.forEach(id => {
        if (id !== chartId) {
          const chartRef = chartRefs.current.get(id)
          if (chartRef && chartRef.setZoomRange) {
            chartRef.setZoomRange(range)
          }
        }
      })
    }
  }

  const breakpoints = { lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }
  const cols = { lg: 4, md: 3, sm: 2, xs: 1, xxs: 1 }

  return (
    <div style={{ height: '100%', position: 'relative' }}>
      {/* Toolbar */}
      <div style={{ 
        padding: '8px 16px', 
        background: theme === 'dark' ? '#1a1a1a' : '#fafafa',
        borderBottom: `1px solid ${theme === 'dark' ? '#333' : '#e8e8e8'}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <Space>
          <Button 
            type="primary" 
            size="small" 
            icon={<PlusOutlined />}
            onClick={() => setIsAddChartModalVisible(true)}
          >
            Add Chart
          </Button>
          
          {selectedCharts.length > 1 && (
            <Button
              size="small"
              icon={<LinkOutlined />}
              onClick={handleCreateSyncGroup}
            >
              Sync Selected ({selectedCharts.length})
            </Button>
          )}
        </Space>

        <Space>
          <span style={{ fontSize: '12px', color: '#666' }}>
            {layout.charts.length} chart{layout.charts.length !== 1 ? 's' : ''}
          </span>
          {synchronizationGroups.length > 0 && (
            <span style={{ fontSize: '12px', color: '#1890ff' }}>
              {synchronizationGroups.length} sync group{synchronizationGroups.length !== 1 ? 's' : ''}
            </span>
          )}
        </Space>
      </div>

      {/* Charts Grid */}
      <div style={{ height: 'calc(100% - 49px)', overflow: 'auto' }}>
        {layout.charts.length === 0 ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: '#999'
          }}>
            <PlusOutlined style={{ fontSize: 48, marginBottom: 16 }} />
            <h3>No Charts Added</h3>
            <p>Click "Add Chart" to start building your layout</p>
          </div>
        ) : (
          <ResponsiveGridLayout
            className="layout"
            layouts={gridLayouts}
            breakpoints={breakpoints}
            cols={cols}
            rowHeight={200}
            onLayoutChange={handleGridLayoutChange}
            onDragStart={() => setIsDragging(true)}
            onDragStop={() => setIsDragging(false)}
            onResizeStart={() => setIsDragging(true)}
            onResizeStop={() => setIsDragging(false)}
            margin={[8, 8]}
            containerPadding={[8, 8]}
            isDraggable={true}
            isResizable={true}
            compactType={null}
            preventCollision={true}
          >
            {layout.charts.map(chart => {
              const data = chartData.get(chart.instrument.symbol) || []
              const isSelected = selectedCharts.includes(chart.id)
              const syncGroup = chartLayoutService.getSyncGroupForChart(chart.id)

              return (
                <div key={chart.id}>
                  <Card
                    size="small"
                    title={
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={(e) => handleChartSelect(chart.id, e.target.checked)}
                          style={{ margin: 0 }}
                        />
                        <DragOutlined style={{ cursor: 'grab', color: '#999' }} />
                        <span style={{ fontSize: '12px' }}>
                          {chart.instrument.symbol} ({chart.chartType})
                        </span>
                        {syncGroup && (
                          <Tooltip title={`Synchronized with ${syncGroup.chartIds.length - 1} other chart(s)`}>
                            <LinkOutlined style={{ color: '#1890ff', fontSize: '12px' }} />
                          </Tooltip>
                        )}
                      </div>
                    }
                    extra={
                      <Space size="small">
                        <Tooltip title="Chart settings">
                          <Button
                            type="text"
                            size="small"
                            icon={<SettingOutlined />}
                            style={{ padding: '2px 4px' }}
                          />
                        </Tooltip>
                        <Tooltip title="Remove chart">
                          <Button
                            type="text"
                            danger
                            size="small"
                            icon={<DeleteOutlined />}
                            onClick={() => handleRemoveChart(chart.id)}
                            style={{ padding: '2px 4px' }}
                          />
                        </Tooltip>
                      </Space>
                    }
                    bodyStyle={{ padding: 0 }}
                    style={{
                      height: '100%',
                      border: isSelected ? '2px solid #1890ff' : undefined
                    }}
                  >
                    <div style={{ height: 'calc(100% - 32px)' }}>
                      <AdvancedChartContainer
                        ref={(ref) => chartRefs.current.set(chart.id, ref)}
                        data={data}
                        chartType={chart.chartType}
                        theme={theme}
                        autoSize={true}
                        onPriceChange={(price) => {
                          // Handle price updates if needed
                        }}
                        indicators={chart.indicators?.map(id => ({
                          id,
                          params: {} // Would need to store params in chart config
                        })) || []}
                      />
                    </div>
                  </Card>
                </div>
              )
            })}
          </ResponsiveGridLayout>
        )}
      </div>

      {/* Add Chart Modal */}
      <Modal
        title="Add New Chart"
        open={isAddChartModalVisible}
        onOk={handleAddChart}
        onCancel={() => setIsAddChartModalVisible(false)}
        width={500}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <label style={{ display: 'block', marginBottom: 8 }}>Instrument:</label>
            <Select
              value={selectedInstrument}
              onChange={setSelectedInstrument}
              style={{ width: '100%' }}
              placeholder="Select an instrument"
              showSearch
              filterOption={(input, option) =>
                option?.children?.toString().toLowerCase().includes(input.toLowerCase())
              }
            >
              {instruments.map(instrument => (
                <Option key={instrument.symbol} value={instrument.symbol}>
                  {instrument.symbol} - {instrument.name}
                </Option>
              ))}
            </Select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: 8 }}>Chart Type:</label>
            <ChartTypeSelector
              selectedType={selectedChartType}
              onTypeChange={setSelectedChartType}
            />
          </div>
        </Space>
      </Modal>
    </div>
  )
}

export default MultiChartView