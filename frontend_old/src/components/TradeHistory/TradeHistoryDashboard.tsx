import React, { useState } from 'react';
import { Tabs, Card, Typography } from 'antd';
import {
  HistoryOutlined,
  BarChartOutlined,
  ExportOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { TradeHistoryTable } from './TradeHistoryTable';

const { Title } = Typography;

interface TabItem {
  key: string;
  label: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}

export const TradeHistoryDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('history');

  const tabItems: TabItem[] = [
    {
      key: 'history',
      label: 'Trade History',
      icon: <HistoryOutlined />,
      children: <TradeHistoryTable />
    },
    {
      key: 'analytics',
      label: 'Performance Analytics',
      icon: <BarChartOutlined />,
      children: (
        <Card>
          <div style={{ textAlign: 'center', padding: '60px 0' }}>
            <BarChartOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
            <Title level={4} style={{ color: '#999', marginTop: 16 }}>
              Performance Analytics
            </Title>
            <p style={{ color: '#666' }}>
              Advanced analytics and charting functionality will be implemented here.
              This will include profit/loss charts, performance metrics, and trading statistics.
            </p>
          </div>
        </Card>
      )
    },
    {
      key: 'reports',
      label: 'Reports & Export',
      icon: <ExportOutlined />,
      children: (
        <Card>
          <div style={{ textAlign: 'center', padding: '60px 0' }}>
            <ExportOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
            <Title level={4} style={{ color: '#999', marginTop: 16 }}>
              Reports & Export
            </Title>
            <p style={{ color: '#666' }}>
              Comprehensive reporting tools will be available here.
              Generate custom reports, tax documents, and export data in various formats.
            </p>
          </div>
        </Card>
      )
    },
    {
      key: 'settings',
      label: 'Settings',
      icon: <SettingOutlined />,
      children: (
        <Card>
          <div style={{ textAlign: 'center', padding: '60px 0' }}>
            <SettingOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
            <Title level={4} style={{ color: '#999', marginTop: 16 }}>
              Trade Settings
            </Title>
            <p style={{ color: '#666' }}>
              Configure trade tracking preferences, strategy classifications,
              and data synchronization settings.
            </p>
          </div>
        </Card>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <HistoryOutlined /> Trade History & Execution Log
        </Title>
        <p style={{ color: '#666', fontSize: '16px' }}>
          View and analyze your complete trade history with detailed execution information,
          performance metrics, and export capabilities.
        </p>
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={tabItems}
        size="large"
        tabBarStyle={{ marginBottom: 24 }}
      />
    </div>
  );
};