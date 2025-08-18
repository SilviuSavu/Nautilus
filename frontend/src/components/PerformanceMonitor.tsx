/**
 * Performance monitoring component for WebSocket communication metrics
 */

import React from 'react'
import { Card, Space, Statistic, Alert, Row, Col, Progress, Tooltip, Typography } from 'antd'
import { TrophyOutlined, ThunderboltOutlined, WarningOutlined } from '@ant-design/icons'
import { PerformanceMetrics } from '../services/websocket'

const { Text, Title } = Typography

interface PerformanceMonitorProps {
  metrics: PerformanceMetrics;
  compact?: boolean;
}

const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({ 
  metrics, 
  compact = false 
}) => {
  // Performance thresholds
  const LATENCY_THRESHOLD_MS = 100
  const CRITICAL_LATENCY_MS = 200
  
  const getLatencyStatus = () => {
    if (metrics.averageLatency <= LATENCY_THRESHOLD_MS) return 'success'
    if (metrics.averageLatency <= CRITICAL_LATENCY_MS) return 'warning'
    return 'error'
  }

  const getLatencyColor = () => {
    const status = getLatencyStatus()
    switch (status) {
      case 'success': return '#52c41a'
      case 'warning': return '#faad14'
      case 'error': return '#ff4d4f'
      default: return '#1890ff'
    }
  }

  const getLatencyMessage = () => {
    const status = getLatencyStatus()
    switch (status) {
      case 'success': return `Latency within requirement (<${LATENCY_THRESHOLD_MS}ms)`
      case 'warning': return `Latency approaching threshold (${LATENCY_THRESHOLD_MS}ms)`
      case 'error': return `Latency exceeds requirement (>${LATENCY_THRESHOLD_MS}ms)`
      default: return 'Measuring latency...'
    }
  }

  const formatLatency = (latency: number) => {
    if (latency === Infinity || latency === 0) return 'N/A'
    return `${latency.toFixed(2)}ms`
  }

  const latencyProgress = Math.min((metrics.averageLatency / CRITICAL_LATENCY_MS) * 100, 100)

  if (compact) {
    return (
      <Alert
        message={
          <Space>
            <ThunderboltOutlined />
            <Text>Avg Latency: {formatLatency(metrics.averageLatency)}</Text>
            {metrics.averageLatency > LATENCY_THRESHOLD_MS && (
              <WarningOutlined style={{ color: '#faad14' }} />
            )}
          </Space>
        }
        type={getLatencyStatus()}
        showIcon
      />
    )
  }

  return (
    <Card
      title={
        <Space>
          <TrophyOutlined />
          <Title level={4} style={{ margin: 0 }}>Performance Metrics</Title>
        </Space>
      }
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        {/* Latency Status Alert */}
        <Alert
          message={getLatencyMessage()}
          type={getLatencyStatus()}
          showIcon
        />

        {/* Performance Statistics */}
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="Average Latency"
              value={formatLatency(metrics.averageLatency)}
              valueStyle={{ color: getLatencyColor() }}
              prefix={<ThunderboltOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Max Latency"
              value={formatLatency(metrics.maxLatency)}
              valueStyle={{ 
                color: metrics.maxLatency > CRITICAL_LATENCY_MS ? '#ff4d4f' : '#666' 
              }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Min Latency"
              value={formatLatency(metrics.minLatency)}
              valueStyle={{ color: '#52c41a' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Messages/sec"
              value={metrics.messagesPerSecond.toFixed(1)}
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
        </Row>

        {/* Latency Progress Bar */}
        <div>
          <Text strong>Latency Performance</Text>
          <Tooltip 
            title={`${formatLatency(metrics.averageLatency)} / ${CRITICAL_LATENCY_MS}ms threshold`}
          >
            <Progress
              percent={latencyProgress}
              status={getLatencyStatus() === 'error' ? 'exception' : 'normal'}
              strokeColor={getLatencyColor()}
              size="small"
            />
          </Tooltip>
        </div>

        {/* Additional Metrics */}
        <Row gutter={16}>
          <Col span={12}>
            <Statistic
              title="Total Messages Processed"
              value={metrics.messagesProcessed}
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="Sample Size"
              value={metrics.messageLatency.length}
              suffix={`/ 100`}
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
        </Row>

        {/* Recent Latency Trend */}
        {metrics.messageLatency.length > 0 && (
          <div>
            <Text strong>Recent Latency Samples:</Text>
            <div style={{ marginTop: 8 }}>
              {metrics.messageLatency.slice(-10).map((latency, index) => (
                <span
                  key={index}
                  style={{
                    display: 'inline-block',
                    margin: '2px',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    backgroundColor: latency > LATENCY_THRESHOLD_MS ? '#fff1f0' : '#f6ffed',
                    border: `1px solid ${latency > LATENCY_THRESHOLD_MS ? '#ffccc7' : '#b7eb8f'}`,
                    fontSize: '12px'
                  }}
                >
                  {latency.toFixed(1)}ms
                </span>
              ))}
            </div>
          </div>
        )}
      </Space>
    </Card>
  )
}

export default PerformanceMonitor