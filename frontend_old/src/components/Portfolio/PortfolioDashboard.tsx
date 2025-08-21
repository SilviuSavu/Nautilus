/**
 * Portfolio Dashboard Component - Main dashboard integrating all portfolio components
 */

import React, { useState } from 'react';
import { Layout, Tabs, Card, Row, Col, Space, Button, Select } from 'antd';
import { 
  DashboardOutlined, 
  PieChartOutlined, 
  BarChartOutlined,
  LineChartOutlined,
  GlobalOutlined,
  SettingOutlined,
  FullscreenOutlined
} from '@ant-design/icons';

// Import all portfolio components
import PortfolioPnLChart from './PortfolioPnLChart';
import StrategyContributionAnalysis from './StrategyContributionAnalysis';
import StrategyComparison from './StrategyComparison';
import StrategyCorrelationMatrix from './StrategyCorrelationMatrix';
import RelativePerformanceChart from './RelativePerformanceChart';
import AssetAllocationChart from './AssetAllocationChart';
import DiversificationAnalysis from './DiversificationAnalysis';
import AllocationDriftMonitor from './AllocationDriftMonitor';

const { Content } = Layout;
const { TabPane } = Tabs;
const { Option } = Select;

interface PortfolioDashboardProps {
  defaultTab?: string;
  timeframe?: '1D' | '1W' | '1M' | '3M' | '6M' | '1Y';
}

const PortfolioDashboard: React.FC<PortfolioDashboardProps> = ({
  defaultTab = 'overview',
  timeframe = '1M'
}) => {
  const [activeTab, setActiveTab] = useState(defaultTab);
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const tabItems = [
    {
      key: 'overview',
      label: (
        <span>
          <DashboardOutlined />
          Portfolio Overview
        </span>
      ),
      children: (
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <PortfolioPnLChart 
              height={350}
              timeframe={selectedTimeframe}
              showStrategyBreakdown={true}
            />
          </Col>
          <Col span={12}>
            <StrategyContributionAnalysis 
              period={selectedTimeframe}
              showAttribution={true}
              maxStrategies={8}
            />
          </Col>
          <Col span={12}>
            <AssetAllocationChart 
              height={350}
              groupBy="asset_class"
              chartType="doughnut"
            />
          </Col>
        </Row>
      ),
    },
    {
      key: 'performance',
      label: (
        <span>
          <LineChartOutlined />
          Performance Analysis
        </span>
      ),
      children: (
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <StrategyComparison 
              period={selectedTimeframe}
              comparison="absolute"
              maxStrategies={10}
            />
          </Col>
          <Col span={24}>
            <RelativePerformanceChart 
              height={400}
              timeframe={selectedTimeframe}
              benchmark="SP500"
            />
          </Col>
        </Row>
      ),
    },
    {
      key: 'correlation',
      label: (
        <span>
          <BarChartOutlined />
          Correlation Analysis
        </span>
      ),
      children: (
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <StrategyCorrelationMatrix 
              timeframe={selectedTimeframe}
              showPValues={false}
              minCorrelation={0.1}
            />
          </Col>
        </Row>
      ),
    },
    {
      key: 'allocation',
      label: (
        <span>
          <PieChartOutlined />
          Asset Allocation
        </span>
      ),
      children: (
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <AssetAllocationChart 
              height={400}
              groupBy="sector"
              chartType="pie"
            />
          </Col>
          <Col span={12}>
            <AssetAllocationChart 
              height={400}
              groupBy="geography"
              chartType="bar"
            />
          </Col>
          <Col span={24}>
            <DiversificationAnalysis analysisType="sector" />
          </Col>
        </Row>
      ),
    },
    {
      key: 'risk',
      label: (
        <span>
          <GlobalOutlined />
          Risk Management
        </span>
      ),
      children: (
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <AllocationDriftMonitor 
              driftThreshold={5}
              analysisType="asset_class"
            />
          </Col>
          <Col span={12}>
            <DiversificationAnalysis analysisType="geography" />
          </Col>
          <Col span={12}>
            <DiversificationAnalysis analysisType="currency" />
          </Col>
        </Row>
      ),
    },
  ];

  const dashboardControls = (
    <Space>
      <Select 
        value={selectedTimeframe} 
        onChange={setSelectedTimeframe}
        style={{ width: 80 }}
      >
        <Option value="1D">1D</Option>
        <Option value="1W">1W</Option>
        <Option value="1M">1M</Option>
        <Option value="3M">3M</Option>
        <Option value="6M">6M</Option>
        <Option value="1Y">1Y</Option>
      </Select>
      
      <Button 
        icon={<FullscreenOutlined />}
        onClick={() => setIsFullscreen(!isFullscreen)}
      >
        {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
      </Button>
      
      <Button icon={<SettingOutlined />}>
        Settings
      </Button>
    </Space>
  );

  return (
    <Layout style={{ 
      minHeight: '100vh',
      padding: isFullscreen ? 0 : '16px',
      backgroundColor: '#f0f2f5'
    }}>
      <Content>
        <Card 
          title="Portfolio Dashboard"
          extra={dashboardControls}
          style={{ 
            minHeight: isFullscreen ? '100vh' : 'auto',
            margin: isFullscreen ? 0 : undefined
          }}
          bodyStyle={{ padding: '16px' }}
        >
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            type="card"
            size="large"
            style={{ minHeight: '600px' }}
            items={tabItems}
          />
        </Card>
      </Content>
    </Layout>
  );
};

export default PortfolioDashboard;