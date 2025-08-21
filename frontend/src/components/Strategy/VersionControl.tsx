import React, { useState, useEffect } from 'react';
import {
  Card,
  List,
  Button,
  Typography,
  Tag,
  Space,
  Modal,
  Descriptions,
  Timeline,
  Alert,
  Spin,
  Tooltip,
  Popconfirm,
  message,
  Row,
  Col,
  Divider,
  Badge
} from 'antd';
import {
  HistoryOutlined,
  RollbackOutlined,
  EyeOutlined,
  DiffOutlined,
  DeleteOutlined,
  CopyOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import {
  StrategyVersion,
  StrategyConfig,
  DeploymentResult,
  VersionComparisonResult
} from './types/strategyTypes';
import { strategyService } from './services/strategyService';

const { Title, Text, Paragraph } = Typography;

interface VersionControlProps {
  strategyId: string;
  currentVersion?: number;
  onVersionSelect?: (version: StrategyVersion) => void;
  onRollback?: (versionId: string) => void;
  visible: boolean;
  onClose: () => void;
}

export const VersionControl: React.FC<VersionControlProps> = ({
  strategyId,
  currentVersion,
  onVersionSelect,
  onRollback,
  visible,
  onClose
}) => {
  const [versions, setVersions] = useState<StrategyVersion[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedVersion, setSelectedVersion] = useState<StrategyVersion | null>(null);
  const [compareVersions, setCompareVersions] = useState<string[]>([]);
  const [comparisonResult, setComparisonResult] = useState<VersionComparisonResult | null>(null);
  const [showComparison, setShowComparison] = useState(false);

  useEffect(() => {
    if (visible) {
      loadVersionHistory();
    }
  }, [visible, strategyId]);

  const loadVersionHistory = async () => {
    setLoading(true);
    try {
      const response = await strategyService.getVersionHistory(strategyId);
      setVersions(response.versions);
    } catch (error) {
      console.error('Failed to load version history:', error);
      message.error('Failed to load version history');
    } finally {
      setLoading(false);
    }
  };

  const handleRollback = async (versionId: string) => {
    try {
      const result = await strategyService.rollbackToVersion(strategyId, versionId);
      if (result.success) {
        message.success('Successfully rolled back to selected version');
        onRollback?.(versionId);
        loadVersionHistory();
        onClose();
      } else {
        message.error(result.error || 'Rollback failed');
      }
    } catch (error) {
      console.error('Rollback failed:', error);
      message.error('Failed to rollback to selected version');
    }
  };

  const handleCompareVersions = async () => {
    if (compareVersions.length !== 2) {
      message.warning('Please select exactly 2 versions to compare');
      return;
    }

    try {
      const result = await strategyService.compareVersions(
        strategyId,
        compareVersions[0],
        compareVersions[1]
      );
      setComparisonResult(result);
      setShowComparison(true);
    } catch (error) {
      console.error('Comparison failed:', error);
      message.error('Failed to compare versions');
    }
  };

  const handleVersionSelection = (versionId: string, checked: boolean) => {
    if (checked) {
      if (compareVersions.length < 2) {
        setCompareVersions([...compareVersions, versionId]);
      }
    } else {
      setCompareVersions(compareVersions.filter(id => id !== versionId));
    }
  };

  const getVersionStatusColor = (version: StrategyVersion) => {
    if (version.version_number === currentVersion) return 'blue';
    if (version.deployment_results?.some(r => r.success)) return 'green';
    if (version.deployment_results?.some(r => !r.success)) return 'red';
    return 'default';
  };

  const getVersionStatusText = (version: StrategyVersion) => {
    if (version.version_number === currentVersion) return 'Current';
    if (version.deployment_results?.some(r => r.success)) return 'Deployed';
    if (version.deployment_results?.some(r => !r.success)) return 'Failed';
    return 'Draft';
  };

  const formatPerformanceMetrics = (results: DeploymentResult[]) => {
    if (!results?.length) return 'No deployment data';
    
    const successful = results.filter(r => r.success);
    if (!successful.length) return 'No successful deployments';

    const totalPnL = successful.reduce((sum, r) => sum + (r.final_pnl?.toNumber() || 0), 0);
    const totalTrades = successful.reduce((sum, r) => sum + r.trade_count, 0);

    return `P&L: $${totalPnL.toFixed(2)}, Trades: ${totalTrades}`;
  };

  const renderVersionItem = (version: StrategyVersion) => (
    <List.Item
      key={version.id}
      actions={[
        <Tooltip title="View Details">
          <Button
            icon={<EyeOutlined />}
            onClick={() => setSelectedVersion(version)}
            size="small"
          />
        </Tooltip>,
        <Tooltip title="Compare">
          <Button
            icon={<DiffOutlined />}
            onClick={() => handleVersionSelection(version.id, !compareVersions.includes(version.id))}
            type={compareVersions.includes(version.id) ? 'primary' : 'default'}
            size="small"
            disabled={!compareVersions.includes(version.id) && compareVersions.length >= 2}
          />
        </Tooltip>,
        version.version_number !== currentVersion && (
          <Popconfirm
            title="Rollback to this version?"
            description="This will replace the current configuration. Are you sure?"
            onConfirm={() => handleRollback(version.id)}
            okText="Rollback"
            cancelText="Cancel"
          >
            <Tooltip title="Rollback">
              <Button
                icon={<RollbackOutlined />}
                danger
                size="small"
              />
            </Tooltip>
          </Popconfirm>
        )
      ].filter(Boolean)}
    >
      <List.Item.Meta
        avatar={
          <Badge count={version.version_number} color="blue">
            <HistoryOutlined style={{ fontSize: 24 }} />
          </Badge>
        }
        title={
          <Space>
            <Text strong>Version {version.version_number}</Text>
            <Tag color={getVersionStatusColor(version)}>
              {getVersionStatusText(version)}
            </Tag>
            {version.deployment_results?.some(r => r.success) && (
              <Tooltip title="Successfully deployed">
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
              </Tooltip>
            )}
          </Space>
        }
        description={
          <div>
            <Paragraph ellipsis={{ rows: 2 }}>
              {version.change_summary}
            </Paragraph>
            <Space split={<Divider type="vertical" />}>
              <Text type="secondary">
                {new Date(version.created_at).toLocaleDateString()}
              </Text>
              <Text type="secondary">
                by {version.created_by}
              </Text>
              <Text type="secondary">
                {formatPerformanceMetrics(version.deployment_results || [])}
              </Text>
            </Space>
          </div>
        }
      />
    </List.Item>
  );

  return (
    <>
      <Modal
        title={
          <Space>
            <HistoryOutlined />
            Strategy Version Control
          </Space>
        }
        open={visible}
        onCancel={onClose}
        width={1000}
        footer={[
          <Button
            key="compare"
            icon={<DiffOutlined />}
            onClick={handleCompareVersions}
            disabled={compareVersions.length !== 2}
          >
            Compare Selected ({compareVersions.length}/2)
          </Button>,
          <Button key="close" onClick={onClose}>
            Close
          </Button>
        ]}
      >
        <Spin spinning={loading}>
          {versions.length > 0 ? (
            <List
              itemLayout="vertical"
              dataSource={versions}
              renderItem={renderVersionItem}
              pagination={{
                pageSize: 5,
                showSizeChanger: false,
                showQuickJumper: true
              }}
            />
          ) : (
            <Alert
              message="No Version History"
              description="This strategy has no saved versions yet. Versions are created when you save configuration changes or deploy the strategy."
              type="info"
              showIcon
            />
          )}
        </Spin>
      </Modal>

      {/* Version Details Modal */}
      <Modal
        title={`Version ${selectedVersion?.version_number} Details`}
        open={!!selectedVersion}
        onCancel={() => setSelectedVersion(null)}
        width={800}
        footer={[
          <Button key="close" onClick={() => setSelectedVersion(null)}>
            Close
          </Button>
        ]}
      >
        {selectedVersion && (
          <div>
            <Descriptions bordered column={2} size="small">
              <Descriptions.Item label="Version" span={1}>
                {selectedVersion.version_number}
              </Descriptions.Item>
              <Descriptions.Item label="Status" span={1}>
                <Tag color={getVersionStatusColor(selectedVersion)}>
                  {getVersionStatusText(selectedVersion)}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Created" span={1}>
                {new Date(selectedVersion.created_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Created By" span={1}>
                {selectedVersion.created_by}
              </Descriptions.Item>
              <Descriptions.Item label="Change Summary" span={2}>
                {selectedVersion.change_summary}
              </Descriptions.Item>
            </Descriptions>

            <Divider orientation="left">Configuration Preview</Divider>
            <Card size="small">
              <pre style={{ maxHeight: 300, overflow: 'auto' }}>
                {JSON.stringify(selectedVersion.config_snapshot, null, 2)}
              </pre>
            </Card>

            {selectedVersion.deployment_results && selectedVersion.deployment_results.length > 0 && (
              <>
                <Divider orientation="left">Deployment History</Divider>
                <Timeline
                  items={selectedVersion.deployment_results.map(result => ({
                    dot: result.success ? 
                      <CheckCircleOutlined style={{ color: '#52c41a' }} /> :
                      <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />,
                    children: (
                      <div>
                        <Space direction="vertical" size="small">
                          <Text strong>
                            {result.success ? 'Successful Deployment' : 'Failed Deployment'}
                          </Text>
                          <Text type="secondary">
                            {new Date(result.start_time).toLocaleString()} - 
                            {result.end_time ? new Date(result.end_time).toLocaleString() : 'Running'}
                          </Text>
                          {result.success && (
                            <Space split={<Divider type="vertical" />}>
                              <Text>P&L: ${result.final_pnl?.toFixed(2) || '0.00'}</Text>
                              <Text>Trades: {result.trade_count}</Text>
                            </Space>
                          )}
                          {result.notes && (
                            <Paragraph type="secondary">{result.notes}</Paragraph>
                          )}
                        </Space>
                      </div>
                    )
                  }))}
                />
              </>
            )}
          </div>
        )}
      </Modal>

      {/* Version Comparison Modal */}
      <Modal
        title="Version Comparison"
        open={showComparison}
        onCancel={() => setShowComparison(false)}
        width={1200}
        footer={[
          <Button key="close" onClick={() => setShowComparison(false)}>
            Close
          </Button>
        ]}
      >
        {comparisonResult && (
          <VersionComparison comparison={comparisonResult} />
        )}
      </Modal>
    </>
  );
};

interface VersionComparisonProps {
  comparison: VersionComparisonResult;
}

const VersionComparison: React.FC<VersionComparisonProps> = ({ comparison }) => {
  const renderDifference = (diff: any) => {
    switch (diff.type) {
      case 'added':
        return <Tag color="green">Added: {diff.path}</Tag>;
      case 'removed':
        return <Tag color="red">Removed: {diff.path}</Tag>;
      case 'changed':
        return (
          <Space direction="vertical" size="small">
            <Tag color="orange">Changed: {diff.path}</Tag>
            <Space>
              <Text delete>{diff.old_value}</Text>
              <Text>→</Text>
              <Text mark>{diff.new_value}</Text>
            </Space>
          </Space>
        );
      default:
        return <Tag>{diff.type}: {diff.path}</Tag>;
    }
  };

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Card title={`Version ${comparison.version1.version_number}`} size="small">
            <Space direction="vertical" size="small">
              <Text type="secondary">
                Created: {new Date(comparison.version1.created_at).toLocaleString()}
              </Text>
              <Text type="secondary">
                By: {comparison.version1.created_by}
              </Text>
              <Paragraph ellipsis={{ rows: 2 }}>
                {comparison.version1.change_summary}
              </Paragraph>
            </Space>
          </Card>
        </Col>
        <Col span={12}>
          <Card title={`Version ${comparison.version2.version_number}`} size="small">
            <Space direction="vertical" size="small">
              <Text type="secondary">
                Created: {new Date(comparison.version2.created_at).toLocaleString()}
              </Text>
              <Text type="secondary">
                By: {comparison.version2.created_by}
              </Text>
              <Paragraph ellipsis={{ rows: 2 }}>
                {comparison.version2.change_summary}
              </Paragraph>
            </Space>
          </Card>
        </Col>
      </Row>

      <Divider orientation="left">Configuration Differences</Divider>
      
      {comparison.differences.length > 0 ? (
        <List
          dataSource={comparison.differences}
          renderItem={(diff, index) => (
            <List.Item key={index}>
              {renderDifference(diff)}
            </List.Item>
          )}
          size="small"
        />
      ) : (
        <Alert
          message="No Differences Found"
          description="The configurations in these versions are identical."
          type="info"
          showIcon
        />
      )}

      {comparison.performance_comparison && (
        <>
          <Divider orientation="left">Performance Comparison</Divider>
          <Row gutter={16}>
            <Col span={12}>
              <Descriptions title="Version 1 Performance" bordered size="small">
                <Descriptions.Item label="Total P&L">
                  ${comparison.performance_comparison.version1_pnl?.toFixed(2) || '0.00'}
                </Descriptions.Item>
                <Descriptions.Item label="Total Trades">
                  {comparison.performance_comparison.version1_trades || 0}
                </Descriptions.Item>
                <Descriptions.Item label="Win Rate">
                  {comparison.performance_comparison.version1_win_rate?.toFixed(2) || 'N/A'}%
                </Descriptions.Item>
              </Descriptions>
            </Col>
            <Col span={12}>
              <Descriptions title="Version 2 Performance" bordered size="small">
                <Descriptions.Item label="Total P&L">
                  ${comparison.performance_comparison.version2_pnl?.toFixed(2) || '0.00'}
                </Descriptions.Item>
                <Descriptions.Item label="Total Trades">
                  {comparison.performance_comparison.version2_trades || 0}
                </Descriptions.Item>
                <Descriptions.Item label="Win Rate">
                  {comparison.performance_comparison.version2_win_rate?.toFixed(2) || 'N/A'}%
                </Descriptions.Item>
              </Descriptions>
            </Col>
          </Row>
        </>
      )}
    </div>
  );
};

export default VersionControl;