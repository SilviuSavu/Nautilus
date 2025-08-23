/**
 * Message Bus Performance Demo Component
 * 
 * Shows the performance difference between:
 * 1. Slow HTTP proxy (current Vite configuration)
 * 2. Fast direct message bus (new high-performance approach)
 */

import React, { useState, useEffect } from 'react';
import { Card, Button, Row, Col, Statistic, Badge, Typography, Space, Divider } from 'antd';
import { ThunderboltOutlined, GlobalOutlined, RocketOutlined, ClockCircleOutlined } from '@ant-design/icons';
import { useMessageBus, useRealtimePortfolio } from '../../hooks/useMessageBus';

const { Title, Text, Paragraph } = Typography;

interface PerformanceMetrics {
  latency: number;
  throughput: number;
  dataPoints: number;
  errors: number;
  startTime: Date;
}

export const MessageBusDemo: React.FC = () => {
  const [httpMetrics, setHttpMetrics] = useState<PerformanceMetrics>({
    latency: 0,
    throughput: 0,
    dataPoints: 0,
    errors: 0,
    startTime: new Date()
  });

  const [messageBusMetrics, setMessageBusMetrics] = useState<PerformanceMetrics>({
    latency: 0,
    throughput: 0,
    dataPoints: 0,
    errors: 0,
    startTime: new Date()
  });

  const [isTestingHttp, setIsTestingHttp] = useState(false);
  const [isTestingMessageBus, setIsTestingMessageBus] = useState(false);

  // Message bus integration
  const { 
    isConnected: messageBusConnected, 
    subscriptions: messageBusSubscriptions,
    reconnectAttempts,
    error: messageBusError,
    sendCommand
  } = useMessageBus();

  // Real-time portfolio via message bus
  const { 
    portfolio: messageBusPortfolio, 
    loading: portfolioLoading,
    error: portfolioError 
  } = useRealtimePortfolio('DU7925702');

  // Test HTTP proxy performance (traditional approach)
  const testHttpPerformance = async () => {
    setIsTestingHttp(true);
    const startTime = Date.now();
    let dataPoints = 0;
    let errors = 0;

    try {
      // Simulate multiple HTTP requests through Vite proxy
      const requests = Array.from({ length: 50 }, async (_, i) => {
        try {
          const start = Date.now();
          const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/realtime/portfolio/DU7925702`);
          const data = await response.json();
          const latency = Date.now() - start;
          dataPoints++;
          return latency;
        } catch (error) {
          errors++;
          return 1000; // High latency for errors
        }
      });

      const latencies = await Promise.all(requests);
      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const totalTime = Date.now() - startTime;
      const throughput = (dataPoints / totalTime) * 1000; // requests per second

      setHttpMetrics({
        latency: Math.round(avgLatency),
        throughput: Math.round(throughput * 100) / 100,
        dataPoints,
        errors,
        startTime: new Date()
      });
    } catch (error) {
      console.error('HTTP test failed:', error);
    } finally {
      setIsTestingHttp(false);
    }
  };

  // Test message bus performance (new approach)
  const testMessageBusPerformance = async () => {
    if (!messageBusConnected) {
      console.warn('Message bus not connected');
      return;
    }

    setIsTestingMessageBus(true);
    const startTime = Date.now();
    let dataPoints = 0;
    let errors = 0;

    try {
      // Simulate multiple message bus commands
      const commands = Array.from({ length: 50 }, async (_, i) => {
        try {
          const start = Date.now();
          const data = await sendCommand('get_realtime_portfolio', { portfolioId: 'DU7925702' });
          const latency = Date.now() - start;
          dataPoints++;
          return latency;
        } catch (error) {
          errors++;
          return 1000; // High latency for errors
        }
      });

      const latencies = await Promise.all(commands);
      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const totalTime = Date.now() - startTime;
      const throughput = (dataPoints / totalTime) * 1000; // commands per second

      setMessageBusMetrics({
        latency: Math.round(avgLatency),
        throughput: Math.round(throughput * 100) / 100,
        dataPoints,
        errors,
        startTime: new Date()
      });
    } catch (error) {
      console.error('Message bus test failed:', error);
    } finally {
      setIsTestingMessageBus(false);
    }
  };

  // Calculate performance improvement
  const latencyImprovement = httpMetrics.latency > 0 && messageBusMetrics.latency > 0 
    ? Math.round(((httpMetrics.latency - messageBusMetrics.latency) / httpMetrics.latency) * 100)
    : 0;

  const throughputImprovement = httpMetrics.throughput > 0 && messageBusMetrics.throughput > 0
    ? Math.round(((messageBusMetrics.throughput - httpMetrics.throughput) / httpMetrics.throughput) * 100)
    : 0;

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <RocketOutlined /> Performance: HTTP Proxy vs Direct Message Bus
      </Title>
      
      <Paragraph>
        <Text strong>Current Issue:</Text> The frontend uses a slow HTTP proxy (Vite dev server) to communicate with the backend, 
        even though we have high-performance Redis message bus and WebSocket infrastructure available.
      </Paragraph>

      <Paragraph>
        <Text strong>Solution:</Text> Direct message bus integration bypasses HTTP proxy entirely for maximum performance.
      </Paragraph>

      <Divider />

      <Row gutter={[24, 24]}>
        {/* HTTP Proxy Performance */}
        <Col span={12}>
          <Card 
            title={
              <Space>
                <GlobalOutlined style={{ color: '#ff4d4f' }} />
                <span>HTTP Proxy (Current - Slow)</span>
                <Badge status="warning" text="Via Vite Proxy" />
              </Space>
            }
            extra={
              <Button 
                type="primary" 
                danger
                onClick={testHttpPerformance}
                loading={isTestingHttp}
                icon={<ClockCircleOutlined />}
              >
                Test HTTP
              </Button>
            }
          >
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="Avg Latency"
                  value={httpMetrics.latency}
                  suffix="ms"
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Throughput"
                  value={httpMetrics.throughput}
                  suffix="req/s"
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Col>
            </Row>

            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={8}>
                <Statistic
                  title="Data Points"
                  value={httpMetrics.dataPoints}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Errors"
                  value={httpMetrics.errors}
                />
              </Col>
              <Col span={8}>
                <Text type="secondary">
                  Path: Frontend → Vite Proxy → Docker Network → Backend
                </Text>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Message Bus Performance */}
        <Col span={12}>
          <Card 
            title={
              <Space>
                <ThunderboltOutlined style={{ color: '#52c41a' }} />
                <span>Direct Message Bus (New - Fast)</span>
                <Badge 
                  status={messageBusConnected ? "success" : "error"} 
                  text={messageBusConnected ? "Connected" : "Disconnected"} 
                />
              </Space>
            }
            extra={
              <Button 
                type="primary"
                onClick={testMessageBusPerformance}
                loading={isTestingMessageBus}
                disabled={!messageBusConnected}
                icon={<ThunderboltOutlined />}
              >
                Test Message Bus
              </Button>
            }
          >
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="Avg Latency"
                  value={messageBusMetrics.latency}
                  suffix="ms"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Throughput"
                  value={messageBusMetrics.throughput}
                  suffix="cmd/s"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
            </Row>

            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={8}>
                <Statistic
                  title="Subscriptions"
                  value={messageBusSubscriptions}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Reconnects"
                  value={reconnectAttempts}
                />
              </Col>
              <Col span={8}>
                <Text type="secondary">
                  Path: Frontend → WebSocket → Redis → Backend
                </Text>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      {/* Performance Comparison */}
      {httpMetrics.latency > 0 && messageBusMetrics.latency > 0 && (
        <Row gutter={24} style={{ marginTop: 24 }}>
          <Col span={24}>
            <Card title="Performance Improvement Analysis">
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title="Latency Improvement"
                    value={latencyImprovement}
                    suffix="%"
                    valueStyle={{ 
                      color: latencyImprovement > 0 ? '#52c41a' : '#ff4d4f' 
                    }}
                    prefix={latencyImprovement > 0 ? '↓' : '↑'}
                  />
                  <Text type="secondary">
                    {latencyImprovement > 0 ? 'Faster' : 'Slower'} response time
                  </Text>
                </Col>
                <Col span={8}>
                  <Statistic
                    title="Throughput Improvement"
                    value={throughputImprovement}
                    suffix="%"
                    valueStyle={{ 
                      color: throughputImprovement > 0 ? '#52c41a' : '#ff4d4f' 
                    }}
                    prefix={throughputImprovement > 0 ? '↑' : '↓'}
                  />
                  <Text type="secondary">
                    {throughputImprovement > 0 ? 'Higher' : 'Lower'} data throughput
                  </Text>
                </Col>
                <Col span={8}>
                  <div>
                    <Title level={4} style={{ margin: 0 }}>
                      {latencyImprovement > 50 ? '🚀 Excellent' : 
                       latencyImprovement > 25 ? '⚡ Good' : 
                       latencyImprovement > 0 ? '📈 Better' : '⚠️ Issue'}
                    </Title>
                    <Text type="secondary">Overall Performance</Text>
                  </div>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      )}

      {/* Real-time Data Demo */}
      <Row gutter={24} style={{ marginTop: 24 }}>
        <Col span={24}>
          <Card title="Real-time Portfolio Data (Via Message Bus)">
            {portfolioLoading && <Text>Loading portfolio via message bus...</Text>}
            {portfolioError && <Text type="danger">Error: {portfolioError}</Text>}
            {messageBusPortfolio && (
              <Row gutter={16}>
                <Col span={6}>
                  <Statistic
                    title="Total Value"
                    value={messageBusPortfolio.total_value}
                    precision={2}
                    prefix="$"
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Total P&L"
                    value={messageBusPortfolio.total_pnl}
                    precision={2}
                    prefix="$"
                    valueStyle={{ 
                      color: messageBusPortfolio.total_pnl >= 0 ? '#52c41a' : '#ff4d4f' 
                    }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Day P&L"
                    value={messageBusPortfolio.day_pnl}
                    precision={2}
                    prefix="$"
                    valueStyle={{ 
                      color: messageBusPortfolio.day_pnl >= 0 ? '#52c41a' : '#ff4d4f' 
                    }}
                  />
                </Col>
                <Col span={6}>
                  <div>
                    <Badge status="success" text={messageBusPortfolio.data_source} />
                    <br />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {messageBusPortfolio.performance_note}
                    </Text>
                  </div>
                </Col>
              </Row>
            )}
          </Card>
        </Col>
      </Row>

      {/* Error States */}
      {messageBusError && (
        <Row gutter={24} style={{ marginTop: 16 }}>
          <Col span={24}>
            <Card>
              <Text type="danger">Message Bus Error: {messageBusError}</Text>
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
};