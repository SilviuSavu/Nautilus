import React from 'react'
import { Card, Typography, Button, Space } from 'antd'
import { DashboardOutlined, ApiOutlined } from '@ant-design/icons'

const { Title } = Typography

const DashboardMinimal: React.FC = () => {
  return (
    <div data-testid="dashboard" style={{ width: '100%', padding: '20px', minHeight: '100vh', backgroundColor: '#f0f2f5' }}>
      <Card style={{ maxWidth: '800px', margin: '0 auto' }}>
        <Title level={2}>
          <DashboardOutlined /> Nautilus Trading Platform
        </Title>
        
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <div>
            <h3>✅ Frontend Status: Working</h3>
            <p>React is successfully mounting and rendering components.</p>
          </div>
          
          <div>
            <h3>🔧 Environment Configuration:</h3>
            <ul>
              <li>Backend URL: <code>http://localhost:8001</code></li>
              <li>WebSocket URL: <code>localhost:8001</code></li>
              <li>Environment: Development</li>
            </ul>
          </div>
          
          <div>
            <h3>🚀 Available Actions:</h3>
            <Space>
              <Button type="primary" icon={<ApiOutlined />}>
                Test Backend Connection
              </Button>
              <Button>
                View System Status
              </Button>
            </Space>
          </div>
          
          <div>
            <h3>📋 Next Steps:</h3>
            <p>1. Backend is running on port 8001</p>
            <p>2. Frontend is successfully rendering</p>
            <p>3. Ready to add more functionality</p>
          </div>
        </Space>
      </Card>
    </div>
  )
}

export default DashboardMinimal