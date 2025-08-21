import React from 'react';
import { Card, Typography, Row, Col, Statistic, Alert } from 'antd';
import { MonitorOutlined, ClockCircleOutlined, DatabaseOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

interface SystemPerformanceDashboardProps {
  className?: string;
}

export const SystemPerformanceDashboard: React.FC<SystemPerformanceDashboardProps> = ({
  className
}) => {
  return (
    <div className={className}>
      <Card>
        <div style={{ marginBottom: 24 }}>
          <Title level={3}>
            <MonitorOutlined style={{ marginRight: 8 }} />
            System Performance Dashboard
          </Title>
          <Text type="secondary">
            Monitor system health and performance metrics
          </Text>
        </div>
        
        <Alert
          message="System monitoring coming soon"
          description="System performance monitoring features are currently under development"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />
        
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={8}>
            <Card>
              <Statistic
                title="System Uptime"
                value="--"
                prefix={<ClockCircleOutlined />}
                suffix="hours"
              />
            </Card>
          </Col>
          
          <Col xs={24} sm={8}>
            <Card>
              <Statistic
                title="Memory Usage"
                value="--"
                prefix={<MonitorOutlined />}
                suffix="MB"
              />
            </Card>
          </Col>
          
          <Col xs={24} sm={8}>
            <Card>
              <Statistic
                title="Database Size"
                value="--"
                prefix={<DatabaseOutlined />}
                suffix="GB"
              />
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );
};
