import React from 'react'
import { Typography } from 'antd'

const { Title } = Typography

const Dashboard: React.FC = () => {
  return (
    <div style={{ padding: '24px' }}>
      <Title level={1}>🌊 Nautilus Trading Dashboard</Title>
      <p>Dashboard is loading successfully!</p>
      <div data-testid="dashboard">
        <h2>System Status: Online</h2>
        <p>Frontend: ✅ Working</p>
        <p>React: ✅ Rendering</p>
      </div>
    </div>
  )
}

export default Dashboard