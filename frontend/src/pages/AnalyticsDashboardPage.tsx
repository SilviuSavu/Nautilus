/**
 * Analytics Dashboard Page - Integration Example
 * Shows how to use the RealTimeAnalyticsDashboard component in a full page layout
 */

import React, { useState } from 'react';
import { Layout, Card, Row, Col, Select, Button, Space, Typography, Breadcrumb } from 'antd';
import {
  DashboardOutlined,
  SettingOutlined,
  FullscreenOutlined,
  HomeOutlined
} from '@ant-design/icons';
import { RealTimeAnalyticsDashboard } from '../components/Performance';

const { Content } = Layout;
const { Title } = Typography;
const { Option } = Select;

interface AnalyticsDashboardPageProps {
  className?: string;
}

const AnalyticsDashboardPage: React.FC<AnalyticsDashboardPageProps> = ({
  className
}) => {
  const [selectedPortfolio, setSelectedPortfolio] = useState<string>('main-portfolio');
  const [dashboardMode, setDashboardMode] = useState<'full' | 'compact'>('full');
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Mock portfolio options
  const portfolioOptions = [
    { value: 'main-portfolio', label: 'Main Portfolio' },
    { value: 'aggressive-growth', label: 'Aggressive Growth' },
    { value: 'conservative', label: 'Conservative Portfolio' },
    { value: 'hedge-fund-alpha', label: 'Hedge Fund Alpha' },
    { value: 'quantitative-strategies', label: 'Quantitative Strategies' }
  ];

  const handleFullscreenToggle = () => {
    setIsFullscreen(!isFullscreen);
    
    if (!isFullscreen) {
      // Request fullscreen
      if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen();
      }
    } else {
      // Exit fullscreen
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  };

  return (
    <Layout className={className} style={{ minHeight: '100vh' }}>
      <Content style={{ padding: isFullscreen ? '0' : '24px' }}>
        {!isFullscreen && (
          <>
            {/* Breadcrumb Navigation */}
            <Breadcrumb style={{ marginBottom: 16 }}>
              <Breadcrumb.Item>
                <HomeOutlined />
              </Breadcrumb.Item>
              <Breadcrumb.Item>
                <DashboardOutlined />
                <span>Analytics</span>
              </Breadcrumb.Item>
              <Breadcrumb.Item>Real-time Dashboard</Breadcrumb.Item>
            </Breadcrumb>

            {/* Page Header */}
            <Card 
              size="small" 
              style={{ marginBottom: 24 }}
              bodyStyle={{ padding: '16px 24px' }}
            >
              <Row align="middle" justify="space-between">
                <Col>
                  <Space align="center">
                    <DashboardOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
                    <Title level={2} style={{ margin: 0, color: '#262626' }}>
                      Real-time Analytics Dashboard
                    </Title>
                  </Space>
                </Col>
                
                <Col>
                  <Space>
                    <Select
                      value={selectedPortfolio}
                      onChange={setSelectedPortfolio}
                      style={{ width: 200 }}
                      placeholder="Select Portfolio"
                    >
                      {portfolioOptions.map(option => (
                        <Option key={option.value} value={option.value}>
                          {option.label}
                        </Option>
                      ))}
                    </Select>

                    <Select
                      value={dashboardMode}
                      onChange={setDashboardMode}
                      style={{ width: 120 }}
                    >
                      <Option value="full">Full View</Option>
                      <Option value="compact">Compact</Option>
                    </Select>

                    <Button
                      icon={<FullscreenOutlined />}
                      onClick={handleFullscreenToggle}
                      title="Toggle Fullscreen"
                    />
                    
                    <Button
                      icon={<SettingOutlined />}
                      title="Dashboard Settings"
                    />
                  </Space>
                </Col>
              </Row>
            </Card>
          </>
        )}

        {/* Main Dashboard Component */}
        <div style={{ 
          background: isFullscreen ? '#000' : 'transparent',
          padding: isFullscreen ? '16px' : '0',
          minHeight: isFullscreen ? '100vh' : 'auto'
        }}>
          <RealTimeAnalyticsDashboard
            portfolioId={selectedPortfolio}
            compactMode={dashboardMode === 'compact'}
            showStreaming={true}
            updateInterval={250} // 250ms for high-frequency updates
            enableExports={true}
            className={isFullscreen ? 'fullscreen-dashboard' : ''}
          />
        </div>

        {/* Footer Information (hidden in fullscreen) */}
        {!isFullscreen && (
          <Card 
            size="small" 
            style={{ 
              marginTop: 24,
              background: '#fafafa',
              borderColor: '#e8e8e8'
            }}
          >
            <Row justify="space-between" align="middle">
              <Col>
                <Space>
                  <Typography.Text type="secondary">
                    Portfolio: {portfolioOptions.find(p => p.value === selectedPortfolio)?.label}
                  </Typography.Text>
                  <Typography.Text type="secondary">|</Typography.Text>
                  <Typography.Text type="secondary">
                    Update Frequency: 250ms (Sub-second)
                  </Typography.Text>
                  <Typography.Text type="secondary">|</Typography.Text>
                  <Typography.Text type="secondary">
                    Data Sources: IBKR, Alpha Vantage, FRED, EDGAR
                  </Typography.Text>
                </Space>
              </Col>
              
              <Col>
                <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                  Nautilus Trading Platform • Sprint 3 • Real-time Analytics Engine
                </Typography.Text>
              </Col>
            </Row>
          </Card>
        )}
      </Content>

      <style>{`
        .fullscreen-dashboard {
          height: 100vh;
          overflow-y: auto;
        }

        .fullscreen-dashboard .ant-card {
          background: #001529 !important;
          border-color: #303030 !important;
          color: #fff !important;
        }

        .fullscreen-dashboard .ant-typography {
          color: #fff !important;
        }

        .fullscreen-dashboard .ant-statistic-title {
          color: #888 !important;
        }

        .fullscreen-dashboard .ant-statistic-content {
          color: #fff !important;
        }

        /* Custom scrollbar for better aesthetics */
        ::-webkit-scrollbar {
          width: 6px;
        }

        ::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb {
          background: #888;
          border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: #555;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
          .ant-col {
            margin-bottom: 16px;
          }
          
          .ant-space {
            flex-wrap: wrap;
          }
        }

        /* Animation for smooth transitions */
        .ant-card {
          transition: all 0.3s ease;
        }

        .ant-card:hover {
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        /* Performance optimization for charts */
        .ant-spin-container {
          transition: opacity 0.2s ease;
        }
      `}</style>
    </Layout>
  );
};

export default AnalyticsDashboardPage;