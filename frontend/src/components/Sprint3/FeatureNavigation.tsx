import React, { useState } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Typography,
  Badge,
  Tag,
  Tooltip,
  Dropdown,
  Divider,
  List,
  Avatar
} from 'antd';
import {
  WifiOutlined,
  LineChartOutlined,
  SafetyCertificateOutlined,
  RocketOutlined,
  MonitorOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  AlertOutlined,
  DeploymentUnitOutlined,
  ApiOutlined,
  DashboardOutlined,
  ControlOutlined,
  ExperimentOutlined,
  StarOutlined,
  RightOutlined,
  BulbOutlined,
  TrophyOutlined
} from '@ant-design/icons';
import type { Sprint3Feature, QuickAction } from '../../types/sprint3';

const { Title, Text, Paragraph } = Typography;

interface FeatureNavigationProps {
  onFeatureClick: (featureId: string) => void;
  onQuickAction?: (actionId: string) => void;
  showQuickActions?: boolean;
  compactMode?: boolean;
}

const FeatureNavigation: React.FC<FeatureNavigationProps> = ({
  onFeatureClick,
  onQuickAction,
  showQuickActions = true,
  compactMode = false
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  // Sprint 3 Features
  const features: Sprint3Feature[] = [
    // WebSocket Infrastructure
    {
      id: 'websocket-monitor',
      name: 'WebSocket Monitor',
      description: 'Real-time WebSocket connection monitoring and diagnostics',
      icon: 'WifiOutlined',
      path: '/sprint3/websocket',
      category: 'websocket',
      status: 'active',
      dependencies: ['messagebus'],
      permissions: ['view_infrastructure']
    },
    {
      id: 'message-streaming',
      name: 'Message Streaming',
      description: 'High-performance message streaming and routing',
      icon: 'ThunderboltOutlined',
      path: '/sprint3/messaging',
      category: 'websocket',
      status: 'active',
      dependencies: ['websocket-monitor'],
      permissions: ['manage_messaging']
    },

    // Real-time Analytics
    {
      id: 'realtime-dashboard',
      name: 'Real-time Dashboard',
      description: 'Live performance metrics and analytics streaming',
      icon: 'LineChartOutlined',
      path: '/sprint3/analytics',
      category: 'analytics',
      status: 'active',
      dependencies: ['websocket-monitor'],
      permissions: ['view_analytics']
    },
    {
      id: 'metric-streaming',
      name: 'Metric Streaming',
      description: 'Continuous performance metric calculation and distribution',
      icon: 'DashboardOutlined',
      path: '/sprint3/metrics',
      category: 'analytics',
      status: 'active',
      dependencies: ['realtime-dashboard'],
      permissions: ['manage_metrics']
    },
    {
      id: 'alert-engine',
      name: 'Alert Engine',
      description: 'Dynamic threshold monitoring and intelligent alerting',
      icon: 'AlertOutlined',
      path: '/sprint3/alerts',
      category: 'analytics',
      status: 'beta',
      dependencies: ['metric-streaming'],
      permissions: ['manage_alerts']
    },

    // Risk Management
    {
      id: 'dynamic-limits',
      name: 'Dynamic Risk Limits',
      description: 'Adaptive risk limit management with real-time adjustments',
      icon: 'SafetyCertificateOutlined',
      path: '/sprint3/risk-limits',
      category: 'risk',
      status: 'active',
      dependencies: ['realtime-dashboard'],
      permissions: ['manage_risk']
    },
    {
      id: 'breach-detection',
      name: 'Breach Detection',
      description: 'Automated risk breach detection and response system',
      icon: 'ExperimentOutlined',
      path: '/sprint3/breach-detection',
      category: 'risk',
      status: 'active',
      dependencies: ['dynamic-limits'],
      permissions: ['manage_risk']
    },
    {
      id: 'risk-analytics',
      name: 'Risk Analytics',
      description: 'Advanced risk analytics and scenario modeling',
      icon: 'TrophyOutlined',
      path: '/sprint3/risk-analytics',
      category: 'risk',
      status: 'beta',
      dependencies: ['breach-detection'],
      permissions: ['view_risk_analytics']
    },

    // Strategy Deployment
    {
      id: 'deployment-pipeline',
      name: 'Deployment Pipeline',
      description: 'Automated strategy deployment with approval workflows',
      icon: 'RocketOutlined',
      path: '/sprint3/deployment',
      category: 'strategy',
      status: 'active',
      dependencies: ['dynamic-limits'],
      permissions: ['deploy_strategies']
    },
    {
      id: 'testing-framework',
      name: 'Testing Framework',
      description: 'Comprehensive strategy testing and validation suite',
      icon: 'ExperimentOutlined',
      path: '/sprint3/testing',
      category: 'strategy',
      status: 'active',
      dependencies: ['deployment-pipeline'],
      permissions: ['manage_testing']
    },
    {
      id: 'rollback-system',
      name: 'Rollback System',
      description: 'Intelligent rollback and recovery mechanisms',
      icon: 'ControlOutlined',
      path: '/sprint3/rollback',
      category: 'strategy',
      status: 'beta',
      dependencies: ['testing-framework'],
      permissions: ['manage_rollback']
    },

    // System Monitoring
    {
      id: 'system-health',
      name: 'System Health',
      description: 'Comprehensive system health monitoring and diagnostics',
      icon: 'MonitorOutlined',
      path: '/sprint3/system-health',
      category: 'monitoring',
      status: 'active',
      dependencies: [],
      permissions: ['view_system_health']
    },
    {
      id: 'performance-profiler',
      name: 'Performance Profiler',
      description: 'Deep performance analysis and optimization insights',
      icon: 'DashboardOutlined',
      path: '/sprint3/profiler',
      category: 'monitoring',
      status: 'experimental',
      dependencies: ['system-health'],
      permissions: ['manage_profiler']
    }
  ];

  // Quick Actions
  const quickActions: QuickAction[] = [
    {
      id: 'restart-websockets',
      name: 'Restart WebSockets',
      description: 'Restart all WebSocket connections',
      icon: 'WifiOutlined',
      category: 'system',
      action: async () => {
        console.log('Restarting WebSocket connections...');
        // Implementation would go here
      }
    },
    {
      id: 'refresh-metrics',
      name: 'Refresh Metrics',
      description: 'Force refresh all real-time metrics',
      icon: 'LineChartOutlined',
      category: 'monitoring',
      action: async () => {
        console.log('Refreshing metrics...');
        // Implementation would go here
      }
    },
    {
      id: 'check-risk-limits',
      name: 'Check Risk Limits',
      description: 'Perform risk limit validation',
      icon: 'SafetyCertificateOutlined',
      category: 'trading',
      action: async () => {
        console.log('Checking risk limits...');
        // Implementation would go here
      }
    },
    {
      id: 'deploy-strategy',
      name: 'Deploy Strategy',
      description: 'Start new strategy deployment',
      icon: 'RocketOutlined',
      category: 'trading',
      action: async () => {
        console.log('Starting strategy deployment...');
        // Implementation would go here
      }
    },
    {
      id: 'system-health-check',
      name: 'Health Check',
      description: 'Run comprehensive system health check',
      icon: 'MonitorOutlined',
      category: 'system',
      action: async () => {
        console.log('Running health check...');
        // Implementation would go here
      }
    },
    {
      id: 'export-logs',
      name: 'Export Logs',
      description: 'Export system logs for analysis',
      icon: 'DatabaseOutlined',
      category: 'configuration',
      action: async () => {
        console.log('Exporting logs...');
        // Implementation would go here
      }
    }
  ];

  // Get icon component from string name
  const getIcon = (iconName: string) => {
    const iconMap: Record<string, React.ReactNode> = {
      WifiOutlined: <WifiOutlined />,
      ThunderboltOutlined: <ThunderboltOutlined />,
      LineChartOutlined: <LineChartOutlined />,
      DashboardOutlined: <DashboardOutlined />,
      AlertOutlined: <AlertOutlined />,
      SafetyCertificateOutlined: <SafetyCertificateOutlined />,
      ExperimentOutlined: <ExperimentOutlined />,
      TrophyOutlined: <TrophyOutlined />,
      RocketOutlined: <RocketOutlined />,
      ControlOutlined: <ControlOutlined />,
      MonitorOutlined: <MonitorOutlined />,
      DatabaseOutlined: <DatabaseOutlined />
    };
    return iconMap[iconName] || <BulbOutlined />;
  };

  // Get status color
  const getStatusColor = (status: Sprint3Feature['status']) => {
    switch (status) {
      case 'active': return 'success';
      case 'beta': return 'processing';
      case 'experimental': return 'warning';
      default: return 'default';
    }
  };

  // Filter features by category
  const filteredFeatures = selectedCategory === 'all' 
    ? features 
    : features.filter(f => f.category === selectedCategory);

  // Category options
  const categories = [
    { key: 'all', label: 'All Features', icon: <StarOutlined /> },
    { key: 'websocket', label: 'WebSocket', icon: <WifiOutlined /> },
    { key: 'analytics', label: 'Analytics', icon: <LineChartOutlined /> },
    { key: 'risk', label: 'Risk', icon: <SafetyCertificateOutlined /> },
    { key: 'strategy', label: 'Strategy', icon: <RocketOutlined /> },
    { key: 'monitoring', label: 'Monitoring', icon: <MonitorOutlined /> }
  ];

  if (compactMode) {
    return (
      <Row gutter={[8, 8]}>
        {features.filter(f => f.status === 'active').slice(0, 6).map(feature => (
          <Col xs={12} sm={8} md={4} key={feature.id}>
            <Button
              type="default"
              size="small"
              block
              icon={getIcon(feature.icon)}
              onClick={() => onFeatureClick(feature.id)}
            >
              {feature.name.split(' ')[0]}
            </Button>
          </Col>
        ))}
      </Row>
    );
  }

  return (
    <div style={{ width: '100%' }}>
      {/* Category Filter */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space wrap>
          {categories.map(cat => (
            <Button
              key={cat.key}
              type={selectedCategory === cat.key ? 'primary' : 'default'}
              size="small"
              icon={cat.icon}
              onClick={() => setSelectedCategory(cat.key)}
            >
              {cat.label}
            </Button>
          ))}
        </Space>
      </Card>

      <Row gutter={[16, 16]}>
        {/* Feature Grid */}
        <Col xs={24} lg={showQuickActions ? 16 : 24}>
          <Card
            title={
              <Space>
                <BulbOutlined />
                Sprint 3 Features
                <Badge count={filteredFeatures.length} color="blue" />
              </Space>
            }
          >
            <Row gutter={[12, 12]}>
              {filteredFeatures.map(feature => (
                <Col xs={24} sm={12} xl={8} key={feature.id}>
                  <Card
                    size="small"
                    hoverable
                    onClick={() => onFeatureClick(feature.id)}
                    style={{ cursor: 'pointer', height: '140px' }}
                    bodyStyle={{ padding: '12px' }}
                  >
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div style={{ fontSize: '20px', color: '#1890ff' }}>
                          {getIcon(feature.icon)}
                        </div>
                        <Tag color={getStatusColor(feature.status)} size="small">
                          {feature.status}
                        </Tag>
                      </div>
                      <div>
                        <Text strong style={{ fontSize: '14px' }}>
                          {feature.name}
                        </Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {feature.description}
                        </Text>
                      </div>
                      {feature.dependencies.length > 0 && (
                        <div>
                          <Text type="secondary" style={{ fontSize: '10px' }}>
                            Depends: {feature.dependencies.slice(0, 2).join(', ')}
                            {feature.dependencies.length > 2 && '...'}
                          </Text>
                        </div>
                      )}
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* Quick Actions */}
        {showQuickActions && (
          <Col xs={24} lg={8}>
            <Card
              title={
                <Space>
                  <ThunderboltOutlined />
                  Quick Actions
                </Space>
              }
              size="small"
            >
              <List
                size="small"
                dataSource={quickActions}
                renderItem={(action) => (
                  <List.Item
                    style={{ 
                      cursor: 'pointer',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      marginBottom: '4px'
                    }}
                    className="quick-action-item"
                    onClick={() => {
                      action.action();
                      onQuickAction?.(action.id);
                    }}
                  >
                    <List.Item.Meta
                      avatar={
                        <Avatar
                          size="small"
                          icon={getIcon(action.icon)}
                          style={{ backgroundColor: '#1890ff' }}
                        />
                      }
                      title={
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Text strong style={{ fontSize: '13px' }}>
                            {action.name}
                          </Text>
                          <RightOutlined style={{ fontSize: '10px', color: '#999' }} />
                        </div>
                      }
                      description={
                        <Text type="secondary" style={{ fontSize: '11px' }}>
                          {action.description}
                        </Text>
                      }
                    />
                  </List.Item>
                )}
              />
            </Card>
          </Col>
        )}
      </Row>

      {/* Feature Stats */}
      <Card size="small" style={{ marginTop: 16 }}>
        <Row gutter={[16, 8]} align="middle">
          <Col span={6}>
            <Tooltip title="Features ready for production use">
              <Statistic
                title="Active"
                value={features.filter(f => f.status === 'active').length}
                prefix={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
                valueStyle={{ fontSize: '16px' }}
              />
            </Tooltip>
          </Col>
          <Col span={6}>
            <Tooltip title="Features in beta testing">
              <Statistic
                title="Beta"
                value={features.filter(f => f.status === 'beta').length}
                prefix={<ExperimentOutlined style={{ color: '#1890ff' }} />}
                valueStyle={{ fontSize: '16px' }}
              />
            </Tooltip>
          </Col>
          <Col span={6}>
            <Tooltip title="Experimental features under development">
              <Statistic
                title="Experimental"
                value={features.filter(f => f.status === 'experimental').length}
                prefix={<BulbOutlined style={{ color: '#faad14' }} />}
                valueStyle={{ fontSize: '16px' }}
              />
            </Tooltip>
          </Col>
          <Col span={6}>
            <Tooltip title="Total Sprint 3 features available">
              <Statistic
                title="Total"
                value={features.length}
                prefix={<StarOutlined style={{ color: '#722ed1' }} />}
                valueStyle={{ fontSize: '16px' }}
              />
            </Tooltip>
          </Col>
        </Row>
      </Card>

      <style jsx>{`
        .quick-action-item:hover {
          background-color: #f0f0f0;
        }
      `}</style>
    </div>
  );
};

export default FeatureNavigation;