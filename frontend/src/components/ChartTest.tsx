import React, { useEffect, useRef } from 'react'
import { createChart, IChartApi } from 'lightweight-charts'
import { Button, Space, Card, Divider } from 'antd'
import { ChartComponent } from './Chart'
import { useChartStore } from './Chart/hooks/useChartStore'

const ChartTest: React.FC = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const { setCurrentInstrument, currentInstrument } = useChartStore()

  // Test instrument
  const testInstrument = {
    symbol: 'AAPL',
    venue: 'NASDAQ', 
    assetClass: 'STK',
    currency: 'USD',
    id: 'AAPL-STK-test',
    name: 'Apple Inc. (Test)'
  }

  const handleTestInstrumentSelection = () => {
    console.log('🎯 Setting test instrument for chart:', testInstrument)
    setCurrentInstrument(testInstrument)
  }

  // Initialize native TradingView test chart
  useEffect(() => {
    console.log('🚀 Initializing native TradingView chart...')
    
    if (!chartContainerRef.current) {
      return
    }

    const container = chartContainerRef.current
    
    // Wait for container to have proper dimensions
    const initChart = () => {
      if (container.clientWidth === 0 || container.clientHeight === 0) {
        setTimeout(initChart, 100)
        return
      }

      try {
        if (chartRef.current) {
          chartRef.current.remove()
        }

        const chart = createChart(container, {
          width: container.clientWidth,
          height: container.clientHeight,
          layout: {
            background: { color: '#ffffff' },
            textColor: '#333',
          },
          grid: {
            vertLines: { color: '#e1e1e1' },
            horzLines: { color: '#e1e1e1' },
          },
        })

        const candlestickSeries = chart.addCandlestickSeries({
          upColor: '#26a69a',
          downColor: '#ef5350',
        })

        // Simple test data
        const mockData = [
          { time: '2024-08-01', open: 150, high: 155, low: 148, close: 152 },
          { time: '2024-08-02', open: 152, high: 158, low: 151, close: 156 },
          { time: '2024-08-03', open: 156, high: 159, low: 154, close: 157 },
          { time: '2024-08-04', open: 157, high: 161, low: 156, close: 160 },
          { time: '2024-08-05', open: 160, high: 162, low: 158, close: 159 }
        ]

        candlestickSeries.setData(mockData)
        chart.timeScale().fitContent()
        chartRef.current = chart

        console.log('✅ Native TradingView chart initialized successfully')
      } catch (error) {
        console.error('❌ Native chart initialization failed:', error)
      }
    }

    initChart()

    return () => {
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [])

  return (
    <div style={{ padding: '20px' }}>
      <h2>📊 TradingView Chart Diagnostics</h2>
      
      <Card title="1. Native TradingView Test" style={{ marginBottom: '20px' }}>
        <p>Direct TradingView Lightweight Charts implementation with mock data.</p>
        <div 
          ref={chartContainerRef}
          style={{ 
            width: '100%', 
            height: '300px',
            border: '2px solid #52c41a',
            backgroundColor: '#ffffff',
            borderRadius: '4px'
          }}
        />
      </Card>

      <Card title="2. Our Chart Component Test">
        <p>
          Test our ChartComponent with API integration. 
          Current instrument: <strong>{currentInstrument?.symbol || 'None'}</strong>
        </p>
        <Space style={{ marginBottom: '16px' }}>
          <Button type="primary" onClick={handleTestInstrumentSelection}>
            Load AAPL Test Data
          </Button>
        </Space>
        
        <div style={{ border: '2px solid #1890ff', borderRadius: '4px', padding: '8px' }}>
          <ChartComponent height={300} />
        </div>
      </Card>
    </div>
  )
}

export default ChartTest