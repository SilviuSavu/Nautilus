import React from 'react'
import { Card, Typography } from 'antd'

const { Title } = Typography

const DashboardSimple: React.FC = () => {
  return (
    <div data-testid="dashboard" style={{ width: '100%', padding: '20px' }}>
      <Card>
        <Title level={2}>Nautilus Dashboard</Title>
        <p>✅ Frontend is working! React is rendering successfully.</p>
        <p>🔧 Troubleshooting complete - all icon import issues resolved.</p>
      </Card>
    </div>
  )
}

export default DashboardSimple