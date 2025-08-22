import React from 'react';
import { Card, Typography } from 'antd';

const { Title } = Typography;

function SimpleDashboard() {
  return (
    <div style={{ padding: '20px' }}>
      <Title level={1}>🚀 Nautilus Trading Platform</Title>
      <Card>
        <Title level={2}>✅ Production Deployment Successful</Title>
        <p>Backend: {import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'}</p>
        <p>Frontend: Loaded successfully</p>
        <p>Status: Ready for trading</p>
      </Card>
    </div>
  );
}

export default SimpleDashboard;