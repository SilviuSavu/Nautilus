import React from 'react';
import { Card, Typography, Button, Space, Alert } from 'antd';
import { DownloadOutlined, FileTextOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

interface DataExportDashboardProps {
  className?: string;
}

export const DataExportDashboard: React.FC<DataExportDashboardProps> = ({
  className
}) => {
  return (
    <div className={className}>
      <Card>
        <div style={{ textAlign: 'center', padding: '40px 20px' }}>
          <FileTextOutlined style={{ fontSize: 48, color: '#1890ff', marginBottom: 16 }} />
          <Title level={3}>Data Export Dashboard</Title>
          <Text type="secondary">
            Export performance data and reports
          </Text>
          
          <div style={{ marginTop: 24 }}>
            <Space direction="vertical">
              <Alert
                message="Export functionality coming soon"
                description="Data export features are currently under development"
                type="info"
                showIcon
              />
              
              <Space>
                <Button type="primary" icon={<DownloadOutlined />} disabled>
                  Export Performance Data
                </Button>
                <Button icon={<DownloadOutlined />} disabled>
                  Export Trade History
                </Button>
              </Space>
            </Space>
          </div>
        </div>
      </Card>
    </div>
  );
};
