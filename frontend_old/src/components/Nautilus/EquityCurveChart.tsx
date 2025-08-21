import React, { useMemo, useState } from 'react'
import { Card, Select, Switch, Row, Col, Statistic, Space, Typography, Tooltip } from 'antd'
import { Line } from '@ant-design/plots'
import { InfoCircleOutlined, ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons'
import dayjs from 'dayjs'
import { EquityPoint, PerformanceMetrics } from '../../services/backtestService'

const { Title, Text } = Typography

interface EquityCurveChartProps {
  data: EquityPoint[]
  metrics?: PerformanceMetrics
  height?: number
  showDrawdown?: boolean
  showBenchmark?: boolean
  benchmarkData?: EquityPoint[]
}

type ChartMode = 'equity' | 'returns' | 'drawdown' | 'combined'

const EquityCurveChart: React.FC<EquityCurveChartProps> = ({
  data,
  metrics,
  height = 400,
  showDrawdown = true,
  showBenchmark = false,
  benchmarkData
}) => {
  const [chartMode, setChartMode] = useState<ChartMode>('equity')
  const [showLogScale, setShowLogScale] = useState(false)
  const [showVolatility, setShowVolatility] = useState(false)

  // Process data for different chart modes
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return []

    const processedData: any[] = []
    
    data.forEach((point, index) => {
      const timestamp = point.timestamp
      const equity = point.equity
      const drawdown = point.drawdown
      
      // Calculate daily returns
      const prevEquity = index > 0 ? data[index - 1].equity : equity
      const dailyReturn = prevEquity > 0 ? ((equity - prevEquity) / prevEquity) * 100 : 0
      
      // Calculate cumulative return
      const initialEquity = data[0].equity
      const cumulativeReturn = ((equity - initialEquity) / initialEquity) * 100

      switch (chartMode) {
        case 'equity':
          processedData.push({
            timestamp,
            value: equity,
            type: 'Equity',
            label: `$${equity.toLocaleString()}`
          })
          if (showDrawdown) {
            processedData.push({
              timestamp,
              value: equity * (1 + drawdown / 100),
              type: 'Drawdown',
              label: `${drawdown.toFixed(2)}%`
            })
          }
          break
          
        case 'returns':
          processedData.push({
            timestamp,
            value: cumulativeReturn,
            type: 'Cumulative Return',
            label: `${cumulativeReturn.toFixed(2)}%`
          })
          break
          
        case 'drawdown':
          processedData.push({
            timestamp,
            value: drawdown,
            type: 'Drawdown',
            label: `${drawdown.toFixed(2)}%`
          })
          break
          
        case 'combined':
          processedData.push(
            {
              timestamp,
              value: cumulativeReturn,
              type: 'Returns',
              label: `${cumulativeReturn.toFixed(2)}%`
            },
            {
              timestamp,
              value: drawdown,
              type: 'Drawdown',
              label: `${drawdown.toFixed(2)}%`
            }
          )
          break
      }
    })

    // Add benchmark data if available
    if (showBenchmark && benchmarkData) {
      benchmarkData.forEach(point => {
        const initialBenchmark = benchmarkData[0].equity
        const benchmarkReturn = ((point.equity - initialBenchmark) / initialBenchmark) * 100
        
        processedData.push({
          timestamp: point.timestamp,
          value: chartMode === 'equity' ? point.equity : benchmarkReturn,
          type: 'Benchmark',
          label: chartMode === 'equity' ? 
            `$${point.equity.toLocaleString()}` : 
            `${benchmarkReturn.toFixed(2)}%`
        })
      })
    }

    return processedData
  }, [data, chartMode, showDrawdown, showBenchmark, benchmarkData])

  // Calculate additional statistics
  const additionalStats = useMemo(() => {
    if (!data || data.length < 2) return null

    const returns = data.slice(1).map((point, index) => {
      const prevEquity = data[index].equity
      return prevEquity > 0 ? ((point.equity - prevEquity) / prevEquity) * 100 : 0
    })

    const positiveReturns = returns.filter(r => r > 0)
    const negativeReturns = returns.filter(r => r < 0)
    
    const volatility = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r, 2), 0) / returns.length
    ) * Math.sqrt(252) // Annualized

    const skewness = calculateSkewness(returns)
    const kurtosis = calculateKurtosis(returns)
    
    const maxEquity = Math.max(...data.map(d => d.equity))
    const minDrawdown = Math.min(...data.map(d => d.drawdown))
    
    const totalDays = data.length
    const tradingDays = Math.max(1, totalDays - 1)
    
    return {
      volatility,
      skewness,
      kurtosis,
      positiveMonths: positiveReturns.length,
      negativeMonths: negativeReturns.length,
      avgPositiveReturn: positiveReturns.length > 0 ? 
        positiveReturns.reduce((a, b) => a + b, 0) / positiveReturns.length : 0,
      avgNegativeReturn: negativeReturns.length > 0 ? 
        negativeReturns.reduce((a, b) => a + b, 0) / negativeReturns.length : 0,
      maxEquity,
      minDrawdown,
      totalDays,
      tradingDays
    }
  }, [data])

  const config = {
    data: chartData,
    xField: 'timestamp',
    yField: 'value',
    seriesField: 'type',
    height,
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 1000,
      },
    },
    color: ['#1890ff', '#ff4d4f', '#52c41a', '#fa8c16'],
    xAxis: {
      type: 'time',
      tickCount: 5,
      label: {
        formatter: (value: string) => dayjs(value).format('MMM DD')
      }
    },
    yAxis: {
      type: showLogScale && chartMode === 'equity' ? 'log' : 'linear',
      label: {
        formatter: (value: number) => {
          if (chartMode === 'equity') {
            return `$${(value / 1000).toFixed(0)}K`
          } else {
            return `${value.toFixed(1)}%`
          }
        }
      }
    },
    tooltip: {
      customContent: (title: string, items: any[]) => {
        if (!items || items.length === 0) return ''
        
        const date = dayjs(title).format('MMM DD, YYYY')
        let content = `<div style="padding: 8px;"><strong>${date}</strong><br/>`
        
        items.forEach(item => {
          const color = item.color
          const name = item.data.type
          const label = item.data.label
          content += `<div style="margin: 4px 0;">
            <span style="color: ${color};">●</span> ${name}: <strong>${label}</strong>
          </div>`
        })
        
        content += '</div>'
        return content
      }
    },
    legend: {
      position: 'top-right' as const,
    },
    area: chartMode === 'drawdown' ? {
      color: '#ff4d4f',
      style: {
        fillOpacity: 0.3,
      }
    } : undefined
  }

  return (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Total Return"
              value={metrics?.totalReturn || 0}
              suffix="%"
              precision={2}
              valueStyle={{ 
                color: (metrics?.totalReturn || 0) >= 0 ? '#3f8600' : '#cf1322',
                fontSize: '18px' 
              }}
              prefix={(metrics?.totalReturn || 0) >= 0 ? 
                <ArrowUpOutlined /> : <ArrowDownOutlined />
              }
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Max Drawdown"
              value={metrics?.maxDrawdown || 0}
              suffix="%"
              precision={2}
              valueStyle={{ color: '#cf1322', fontSize: '18px' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title="Sharpe Ratio"
              value={metrics?.sharpeRatio || 0}
              precision={2}
              valueStyle={{ 
                color: (metrics?.sharpeRatio || 0) >= 1 ? '#3f8600' : '#cf1322',
                fontSize: '18px' 
              }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small">
            <Statistic
              title={additionalStats ? "Volatility" : "Win Rate"}
              value={additionalStats ? additionalStats.volatility : (metrics?.winRate || 0)}
              suffix={additionalStats ? "%" : "%"}
              precision={1}
              valueStyle={{ fontSize: '18px' }}
            />
          </Card>
        </Col>
      </Row>

      <Card 
        title={
          <Space>
            <Title level={4} style={{ margin: 0 }}>Performance Chart</Title>
            <Tooltip title="Interactive equity curve with multiple viewing modes">
              <InfoCircleOutlined />
            </Tooltip>
          </Space>
        }
        extra={
          <Space>
            <Select
              value={chartMode}
              onChange={setChartMode}
              style={{ width: 140 }}
            >
              <Select.Option value="equity">Equity Curve</Select.Option>
              <Select.Option value="returns">Returns</Select.Option>
              <Select.Option value="drawdown">Drawdown</Select.Option>
              <Select.Option value="combined">Combined</Select.Option>
            </Select>
            
            {chartMode === 'equity' && (
              <Switch
                checked={showLogScale}
                onChange={setShowLogScale}
                checkedChildren="Log"
                unCheckedChildren="Linear"
                size="small"
              />
            )}
            
            {showBenchmark && (
              <Switch
                checked={showBenchmark}
                onChange={() => {}}
                checkedChildren="Benchmark"
                unCheckedChildren="Strategy"
                size="small"
                disabled
              />
            )}
          </Space>
        }
      >
        <Line {...config} />
        
        {additionalStats && (
          <Row gutter={[16, 8]} style={{ marginTop: 16, padding: '16px 0' }}>
            <Col xs={24}>
              <Text type="secondary" strong>Additional Statistics:</Text>
            </Col>
            <Col xs={12} sm={8} md={6}>
              <Text type="secondary">Skewness:</Text>
              <div style={{ fontWeight: 500 }}>{additionalStats.skewness.toFixed(2)}</div>
            </Col>
            <Col xs={12} sm={8} md={6}>
              <Text type="secondary">Kurtosis:</Text>
              <div style={{ fontWeight: 500 }}>{additionalStats.kurtosis.toFixed(2)}</div>
            </Col>
            <Col xs={12} sm={8} md={6}>
              <Text type="secondary">Positive Periods:</Text>
              <div style={{ fontWeight: 500, color: '#3f8600' }}>
                {additionalStats.positiveMonths}
              </div>
            </Col>
            <Col xs={12} sm={8} md={6}>
              <Text type="secondary">Negative Periods:</Text>
              <div style={{ fontWeight: 500, color: '#cf1322' }}>
                {additionalStats.negativeMonths}
              </div>
            </Col>
            <Col xs={12} sm={8} md={6}>
              <Text type="secondary">Avg Win:</Text>
              <div style={{ fontWeight: 500, color: '#3f8600' }}>
                {additionalStats.avgPositiveReturn.toFixed(2)}%
              </div>
            </Col>
            <Col xs={12} sm={8} md={6}>
              <Text type="secondary">Avg Loss:</Text>
              <div style={{ fontWeight: 500, color: '#cf1322' }}>
                {additionalStats.avgNegativeReturn.toFixed(2)}%
              </div>
            </Col>
          </Row>
        )}
      </Card>
    </div>
  )
}

// Helper functions for statistical calculations
function calculateSkewness(values: number[]): number {
  const n = values.length
  if (n < 3) return 0
  
  const mean = values.reduce((a, b) => a + b, 0) / n
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n
  const stdDev = Math.sqrt(variance)
  
  if (stdDev === 0) return 0
  
  const skewness = values.reduce((sum, val) => sum + Math.pow((val - mean) / stdDev, 3), 0) / n
  return skewness
}

function calculateKurtosis(values: number[]): number {
  const n = values.length
  if (n < 4) return 0
  
  const mean = values.reduce((a, b) => a + b, 0) / n
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n
  const stdDev = Math.sqrt(variance)
  
  if (stdDev === 0) return 0
  
  const kurtosis = values.reduce((sum, val) => sum + Math.pow((val - mean) / stdDev, 4), 0) / n
  return kurtosis - 3 // Excess kurtosis
}

export default EquityCurveChart