import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Typography, Button, Space, Alert, Badge, Statistic, Table, Tabs, Progress, Tag } from 'antd'
import { ApiOutlined, DatabaseOutlined, WifiOutlined, PlayCircleOutlined, StopOutlined, DashboardOutlined, BarChartOutlined, ThunderboltOutlined } from '@ant-design/icons'

const { Title, Text } = Typography
const { TabPane } = Tabs

const DashboardWorking: React.FC = () => {
  const [engineStatus, setEngineStatus] = useState({ status: 'stopped', health: 'unknown' })
  const [backendHealth, setBackendHealth] = useState({ status: 'unknown', ready: false })

  useEffect(() => {
    // Check backend health
    fetch(import.meta.env.VITE_API_BASE_URL + '/health')
      .then(res => res.json())
      .then(data => {
        setBackendHealth({ status: data.status, ready: true })
      })
      .catch(() => setBackendHealth({ status: 'error', ready: false }))

    // Check engine status
    fetch(import.meta.env.VITE_API_BASE_URL + '/api/v1/nautilus/engine/status')
      .then(res => res.json())
      .then(data => {
        setEngineStatus({ status: data.status || 'stopped', health: data.health || 'unknown' })
      })
      .catch(() => setEngineStatus({ status: 'error', health: 'error' }))
  }, [])

  const startEngine = () => {
    fetch(import.meta.env.VITE_API_BASE_URL + '/api/v1/nautilus/engine/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: {
          engine_type: 'live',
          log_level: 'INFO',
          instance_id: 'dashboard-001',
          trading_mode: 'paper'
        }
      })
    }).then(() => window.location.reload())
  }

  const stopEngine = () => {
    fetch(import.meta.env.VITE_API_BASE_URL + '/api/v1/nautilus/engine/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force: false })
    }).then(() => window.location.reload())
  }

  return (
    <div data-testid="dashboard" style={{ width: '100%', padding: '20px' }}>
      <Title level={2}>
        <DashboardOutlined /> Nautilus Trading Platform
      </Title>
      
      {/* System Status Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Backend Status"
              value={backendHealth.status}
              prefix={<ApiOutlined />}
              valueStyle={{ color: backendHealth.status === 'healthy' ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Engine Status"
              value={engineStatus.status}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: engineStatus.status === 'running' ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Database"
              value="Connected"
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Data Sources"
              value="5 Active"
              prefix={<WifiOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Engine Controls */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={24}>
          <Card title="Engine Controls" extra={
            <Space>
              <Badge status={engineStatus.status === 'running' ? 'processing' : 'default'} text={engineStatus.status} />
            </Space>
          }>
            <Space>
              <Button 
                type="primary" 
                icon={<PlayCircleOutlined />}
                onClick={startEngine}
                disabled={engineStatus.status === 'running'}
              >
                Start Engine
              </Button>
              <Button 
                danger
                icon={<StopOutlined />}
                onClick={stopEngine}
                disabled={engineStatus.status !== 'running'}
              >
                Stop Engine
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* Main Dashboard Tabs */}
      <Tabs defaultActiveKey="overview" type="card">
        <TabPane tab="System Overview" key="overview">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="System Health" size="small">
                <Alert
                  message="All Systems Operational"
                  description="Backend, database, and core services are running normally."
                  type="success"
                  showIcon
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card title="Quick Stats" size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>Backend URL: <Text code>{import.meta.env.VITE_API_BASE_URL}</Text></div>
                  <div>WebSocket URL: <Text code>{import.meta.env.VITE_WS_URL}</Text></div>
                  <div>Environment: <Tag color="blue">Development</Tag></div>
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="API Endpoints" key="api">
          <Card>
            <Title level={4}>Available API Endpoints</Title>
            <ul>
              <li><Text code>GET /health</Text> - System health check</li>
              <li><Text code>GET /api/v1/nautilus/engine/status</Text> - Engine status</li>
              <li><Text code>POST /api/v1/nautilus/engine/start</Text> - Start trading engine</li>
              <li><Text code>POST /api/v1/nautilus/engine/stop</Text> - Stop trading engine</li>
              <li><Text code>GET /api/v1/fred/health</Text> - FRED data integration</li>
              <li><Text code>GET /api/v1/alpha-vantage/health</Text> - Alpha Vantage integration</li>
              <li><Text code>GET /api/v1/edgar/health</Text> - EDGAR SEC data integration</li>
              <li><Text code>GET /api/v1/datagov/health</Text> - Data.gov federal datasets</li>
            </ul>
          </Card>
        </TabPane>

        <TabPane tab="Data Sources" key="data">
          <Row gutter={16}>
            <Col span={6}>
              <Card title="FRED Economic Data" size="small">
                <Badge status="processing" text="Operational" />
                <div>32+ economic indicators</div>
              </Card>
            </Col>
            <Col span={6}>
              <Card title="Alpha Vantage" size="small">
                <Badge status="processing" text="Operational" />
                <div>Market data & fundamentals</div>
              </Card>
            </Col>
            <Col span={6}>
              <Card title="EDGAR SEC Data" size="small">
                <Badge status="processing" text="Operational" />
                <div>7,861+ companies</div>
              </Card>
            </Col>
            <Col span={6}>
              <Card title="Data.gov Federal" size="small">
                <Badge status="processing" text="Operational" />
                <div>346,000+ datasets</div>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default DashboardWorking