import React, { useState, useEffect } from 'react';
import {
  Modal,
  Typography,
  Space,
  Card,
  Row,
  Col,
  Table,
  Tag,
  Descriptions,
  Divider,
  Button,
  Tooltip,
  Alert,
  List,
  Statistic,
  Progress,
  Timeline,
  Tabs,
  Tree,
  Switch,
  Select,
  Input
} from 'antd';
import {
  DiffOutlined,
  SwapOutlined,
  EyeOutlined,
  BarChartOutlined,
  SettingOutlined,
  HistoryOutlined,
  FilterOutlined,
  SearchOutlined
} from '@ant-design/icons';
import {
  StrategyVersion,
  VersionComparisonResult,
  PerformanceComparison,
  ConfigurationDiff,
  ParameterChange
} from './types/strategyTypes';
import { strategyService } from './services/strategyService';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Search } = Input;

interface VersionComparisonProps {
  strategyId: string;
  version1: StrategyVersion;
  version2: StrategyVersion;
  visible: boolean;
  onClose: () => void;
  onApplyChange?: (parameter: string, value: any, targetVersion: number) => void;
}

export const VersionComparison: React.FC<VersionComparisonProps> = ({
  strategyId,
  version1,
  version2,
  visible,
  onClose,
  onApplyChange
}) => {
  const [comparison, setComparison] = useState<VersionComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [showOnlyDifferences, setShowOnlyDifferences] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTab, setActiveTab] = useState('config');

  useEffect(() => {
    if (visible) {
      performComparison();
    }
  }, [visible, version1, version2]);

  const performComparison = async () => {
    setLoading(true);
    try {
      const result = await strategyService.compareVersions(
        strategyId,
        version1.id,
        version2.id
      );
      setComparison(result);
    } catch (error) {
      console.error('Version comparison failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const getDiffTypeColor = (type: string) => {
    switch (type) {
      case 'added': return 'green';
      case 'removed': return 'red';
      case 'modified': return 'orange';
      case 'unchanged': return 'default';
      default: return 'default';
    }
  };

  const getDiffTypeIcon = (type: string) => {
    switch (type) {
      case 'added': return '+';
      case 'removed': return '-';
      case 'modified': return '~';
      case 'unchanged': return '=';
      default: return '?';
    }
  };

  const renderConfigurationDiff = () => {
    if (!comparison?.configuration_diff) return null;

    const filteredDiffs = comparison.configuration_diff.parameter_changes.filter(change => {
      if (showOnlyDifferences && change.change_type === 'unchanged') return false;
      if (selectedCategory !== 'all' && change.category !== selectedCategory) return false;
      if (searchTerm && !change.parameter_name.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      return true;
    });

    const columns = [
      {
        title: 'Parameter',
        dataIndex: 'parameter_name',
        key: 'parameter',
        width: 200,
        render: (name: string, record: ParameterChange) => (
          <Space>
            <Text strong>{name}</Text>
            {record.category && (
              <Tag size="small">{record.category}</Tag>
            )}
          </Space>
        )
      },
      {
        title: 'Change Type',
        dataIndex: 'change_type',
        key: 'change_type',
        width: 120,
        render: (type: string) => (
          <Tag color={getDiffTypeColor(type)}>
            {getDiffTypeIcon(type)} {type.toUpperCase()}
          </Tag>
        )
      },
      {
        title: `Version ${version1.version_number}`,
        dataIndex: 'old_value',
        key: 'old_value',
        render: (value: any, record: ParameterChange) => (
          <div style={{ maxWidth: 200, wordBreak: 'break-all' }}>
            {record.change_type === 'added' ? (
              <Text type="secondary">N/A</Text>
            ) : (
              <Text delete={record.change_type === 'modified'}>
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </Text>
            )}
          </div>
        )
      },
      {
        title: `Version ${version2.version_number}`,
        dataIndex: 'new_value',
        key: 'new_value',
        render: (value: any, record: ParameterChange) => (
          <div style={{ maxWidth: 200, wordBreak: 'break-all' }}>
            {record.change_type === 'removed' ? (
              <Text type="secondary">N/A</Text>
            ) : (
              <Text mark={record.change_type === 'modified'} strong={record.change_type === 'added'}>
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </Text>
            )}
          </div>
        )
      },
      {
        title: 'Impact',
        dataIndex: 'impact_level',
        key: 'impact',
        width: 100,
        render: (impact: string) => (
          <Tag color={impact === 'high' ? 'red' : impact === 'medium' ? 'orange' : 'green'}>
            {impact?.toUpperCase() || 'LOW'}
          </Tag>
        )
      },
      {
        title: 'Actions',
        key: 'actions',
        width: 120,
        render: (_, record: ParameterChange) => (
          <Space>
            {onApplyChange && record.change_type !== 'unchanged' && (
              <Tooltip title={`Apply to Version ${version1.version_number}`}>
                <Button
                  size="small"
                  icon={<SwapOutlined />}
                  onClick={() => onApplyChange(record.parameter_name, record.new_value, version1.version_number)}
                />
              </Tooltip>
            )}
            {record.description && (
              <Tooltip title={record.description}>
                <Button size="small" icon={<EyeOutlined />} />
              </Tooltip>
            )}
          </Space>
        )
      }
    ];

    const categories = [...new Set(comparison.configuration_diff.parameter_changes.map(c => c.category).filter(Boolean))];

    return (
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <Row gutter={16} align="middle">
          <Col>
            <Switch
              checked={showOnlyDifferences}
              onChange={setShowOnlyDifferences}
              checkedChildren="Differences Only"
              unCheckedChildren="Show All"
            />
          </Col>
          <Col>
            <Select
              value={selectedCategory}
              onChange={setSelectedCategory}
              style={{ width: 150 }}
              placeholder="Filter by category"
            >
              <Select.Option value="all">All Categories</Select.Option>
              {categories.map(cat => (
                <Select.Option key={cat} value={cat}>{cat}</Select.Option>
              ))}
            </Select>
          </Col>
          <Col flex="auto">
            <Search
              placeholder="Search parameters..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ maxWidth: 300 }}
              allowClear
            />
          </Col>
        </Row>

        <Alert
          message={`Found ${comparison.configuration_diff.total_changes} changes`}
          description={`${comparison.configuration_diff.high_impact_changes} high impact, ${comparison.configuration_diff.medium_impact_changes} medium impact, ${comparison.configuration_diff.low_impact_changes} low impact`}
          type={comparison.configuration_diff.high_impact_changes > 0 ? 'warning' : 'info'}
          showIcon
        />

        <Table
          columns={columns}
          dataSource={filteredDiffs}
          rowKey="parameter_name"
          size="small"
          scroll={{ y: 400 }}
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showQuickJumper: true
          }}
        />
      </Space>
    );
  };

  const renderPerformanceComparison = () => {
    if (!comparison?.performance_comparison) return null;

    const perf = comparison.performance_comparison;

    return (
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Row gutter={16}>
          <Col span={12}>
            <Card title={`Version ${version1.version_number} Performance`} size="small">
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="Total P&L"
                    value={perf.version1_pnl?.toNumber() || 0}
                    precision={2}
                    prefix="$"
                    valueStyle={{ color: (perf.version1_pnl?.toNumber() || 0) >= 0 ? '#3f8600' : '#cf1322' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Win Rate"
                    value={perf.version1_win_rate || 0}
                    precision={1}
                    suffix="%"
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Total Trades"
                    value={perf.version1_trades || 0}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Sharpe Ratio"
                    value={perf.version1_sharpe || 0}
                    precision={2}
                  />
                </Col>
              </Row>
            </Card>
          </Col>
          <Col span={12}>
            <Card title={`Version ${version2.version_number} Performance`} size="small">
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="Total P&L"
                    value={perf.version2_pnl?.toNumber() || 0}
                    precision={2}
                    prefix="$"
                    valueStyle={{ color: (perf.version2_pnl?.toNumber() || 0) >= 0 ? '#3f8600' : '#cf1322' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Win Rate"
                    value={perf.version2_win_rate || 0}
                    precision={1}
                    suffix="%"
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Total Trades"
                    value={perf.version2_trades || 0}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Sharpe Ratio"
                    value={perf.version2_sharpe || 0}
                    precision={2}
                  />
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>

        <Card title="Performance Delta" size="small">
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="P&L Difference"
                value={perf.pnl_difference?.toNumber() || 0}
                precision={2}
                prefix="$"
                valueStyle={{ color: (perf.pnl_difference?.toNumber() || 0) >= 0 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Win Rate Change"
                value={perf.win_rate_change || 0}
                precision={1}
                suffix="%"
                valueStyle={{ color: (perf.win_rate_change || 0) >= 0 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Trade Count Change"
                value={perf.trade_count_change || 0}
                valueStyle={{ color: (perf.trade_count_change || 0) >= 0 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Sharpe Improvement"
                value={perf.sharpe_improvement || 0}
                precision={3}
                valueStyle={{ color: (perf.sharpe_improvement || 0) >= 0 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
          </Row>

          {perf.statistical_significance && (
            <>
              <Divider />
              <Alert
                message={`Statistical Significance: ${perf.statistical_significance.toFixed(3)}`}
                description={
                  perf.statistical_significance > 0.05 
                    ? "The performance difference is not statistically significant."
                    : "The performance difference is statistically significant."
                }
                type={perf.statistical_significance > 0.05 ? 'warning' : 'success'}
                showIcon
              />
            </>
          )}
        </Card>

        {perf.performance_breakdown && (
          <Card title="Detailed Performance Breakdown" size="small">
            <Table
              columns={[
                { title: 'Metric', dataIndex: 'metric', key: 'metric' },
                { 
                  title: `Version ${version1.version_number}`, 
                  dataIndex: 'version1', 
                  key: 'version1',
                  render: (value: any) => typeof value === 'number' ? value.toFixed(2) : value
                },
                { 
                  title: `Version ${version2.version_number}`, 
                  dataIndex: 'version2', 
                  key: 'version2',
                  render: (value: any) => typeof value === 'number' ? value.toFixed(2) : value
                },
                { 
                  title: 'Change', 
                  dataIndex: 'change', 
                  key: 'change',
                  render: (value: any) => (
                    <Text type={value >= 0 ? 'success' : 'danger'}>
                      {value >= 0 ? '+' : ''}{typeof value === 'number' ? value.toFixed(2) : value}
                    </Text>
                  )
                }
              ]}
              dataSource={perf.performance_breakdown}
              size="small"
              pagination={false}
            />
          </Card>
        )}
      </Space>
    );
  };

  const renderDeploymentHistory = () => {
    const v1Deployments = version1.deployment_results || [];
    const v2Deployments = version2.deployment_results || [];

    return (
      <Row gutter={16}>
        <Col span={12}>
          <Card title={`Version ${version1.version_number} Deployments`} size="small">
            <Timeline
              size="small"
              items={v1Deployments.map(dep => ({
                dot: dep.success ? <CheckCircleOutlined style={{ color: '#52c41a' }} /> : <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />,
                children: (
                  <Space direction="vertical" size="small">
                    <Text strong>{dep.success ? 'Success' : 'Failed'}</Text>
                    <Text type="secondary">
                      {new Date(dep.start_time).toLocaleDateString()}
                    </Text>
                    {dep.final_pnl && (
                      <Text>P&L: ${dep.final_pnl.toFixed(2)}</Text>
                    )}
                    <Text type="secondary">Trades: {dep.trade_count}</Text>
                    {dep.notes && <Paragraph type="secondary">{dep.notes}</Paragraph>}
                  </Space>
                )
              }))}
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card title={`Version ${version2.version_number} Deployments`} size="small">
            <Timeline
              size="small"
              items={v2Deployments.map(dep => ({
                dot: dep.success ? <CheckCircleOutlined style={{ color: '#52c41a' }} /> : <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />,
                children: (
                  <Space direction="vertical" size="small">
                    <Text strong>{dep.success ? 'Success' : 'Failed'}</Text>
                    <Text type="secondary">
                      {new Date(dep.start_time).toLocaleDateString()}
                    </Text>
                    {dep.final_pnl && (
                      <Text>P&L: ${dep.final_pnl.toFixed(2)}</Text>
                    )}
                    <Text type="secondary">Trades: {dep.trade_count}</Text>
                    {dep.notes && <Paragraph type="secondary">{dep.notes}</Paragraph>}
                  </Space>
                )
              }))}
            />
          </Card>
        </Col>
      </Row>
    );
  };

  const renderVersionDetails = () => (
    <Row gutter={16}>
      <Col span={12}>
        <Card title={`Version ${version1.version_number} Details`} size="small">
          <Descriptions size="small" bordered>
            <Descriptions.Item label="Created" span={2}>
              {new Date(version1.created_at).toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label="Created By" span={2}>
              {version1.created_by}
            </Descriptions.Item>
            <Descriptions.Item label="Change Summary" span={2}>
              {version1.change_summary}
            </Descriptions.Item>
          </Descriptions>
        </Card>
      </Col>
      <Col span={12}>
        <Card title={`Version ${version2.version_number} Details`} size="small">
          <Descriptions size="small" bordered>
            <Descriptions.Item label="Created" span={2}>
              {new Date(version2.created_at).toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label="Created By" span={2}>
              {version2.created_by}
            </Descriptions.Item>
            <Descriptions.Item label="Change Summary" span={2}>
              {version2.change_summary}
            </Descriptions.Item>
          </Descriptions>
        </Card>
      </Col>
    </Row>
  );

  return (
    <Modal
      title={
        <Space>
          <DiffOutlined />
          Compare Versions {version1.version_number} & {version2.version_number}
        </Space>
      }
      open={visible}
      onCancel={onClose}
      width={1400}
      footer={[
        <Button key="close" onClick={onClose}>
          Close
        </Button>
      ]}
    >
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="Configuration" key="config" icon={<SettingOutlined />}>
          {renderConfigurationDiff()}
        </TabPane>
        <TabPane tab="Performance" key="performance" icon={<BarChartOutlined />}>
          {renderPerformanceComparison()}
        </TabPane>
        <TabPane tab="Deployment History" key="deployments" icon={<HistoryOutlined />}>
          {renderDeploymentHistory()}
        </TabPane>
        <TabPane tab="Version Details" key="details" icon={<EyeOutlined />}>
          {renderVersionDetails()}
        </TabPane>
      </Tabs>
    </Modal>
  );
};

export default VersionComparison;