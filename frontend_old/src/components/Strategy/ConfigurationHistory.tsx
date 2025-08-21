import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Timeline,
  Typography,
  Space,
  Tag,
  Button,
  Modal,
  Tooltip,
  Alert,
  Spin,
  List,
  Descriptions,
  Divider,
  Row,
  Col,
  Statistic,
  Progress,
  Badge
} from 'antd';
import {
  HistoryOutlined,
  EyeOutlined,
  DiffOutlined,
  SaveOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import {
  ConfigurationChange,
  StrategyConfig,
  ConfigurationSnapshot,
  PerformanceMetrics,
  ConfigurationAudit
} from './types/strategyTypes';
import { strategyService } from './services/strategyService';

const { Title, Text, Paragraph } = Typography;

interface ConfigurationHistoryProps {
  strategyId: string;
  visible: boolean;
  onClose: () => void;
  onRestoreConfig?: (configSnapshot: ConfigurationSnapshot) => void;
}

export const ConfigurationHistory: React.FC<ConfigurationHistoryProps> = ({
  strategyId,
  visible,
  onClose,
  onRestoreConfig
}) => {
  const [history, setHistory] = useState<ConfigurationChange[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedChange, setSelectedChange] = useState<ConfigurationChange | null>(null);
  const [auditLog, setAuditLog] = useState<ConfigurationAudit[]>([]);
  const [performanceHistory, setPerformanceHistory] = useState<PerformanceMetrics[]>([]);

  useEffect(() => {
    if (visible) {
      loadConfigurationHistory();
      loadAuditLog();
      loadPerformanceHistory();
    }
  }, [visible, strategyId]);

  const loadConfigurationHistory = async () => {
    setLoading(true);
    try {
      const response = await strategyService.getConfigurationHistory(strategyId);
      setHistory(response.changes);
    } catch (error) {
      console.error('Failed to load configuration history:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadAuditLog = async () => {
    try {
      const response = await strategyService.getConfigurationAudit(strategyId);
      setAuditLog(response.audit_entries);
    } catch (error) {
      console.error('Failed to load audit log:', error);
    }
  };

  const loadPerformanceHistory = async () => {
    try {
      const response = await strategyService.getPerformanceHistory(strategyId);
      setPerformanceHistory(response.metrics);
    } catch (error) {
      console.error('Failed to load performance history:', error);
    }
  };

  const getChangeTypeIcon = (type: string) => {
    switch (type) {
      case 'parameter_change':
        return <DiffOutlined style={{ color: '#1890ff' }} />;
      case 'deployment':
        return <PlayCircleOutlined style={{ color: '#52c41a' }} />;
      case 'pause':
        return <PauseCircleOutlined style={{ color: '#faad14' }} />;
      case 'stop':
        return <StopOutlined style={{ color: '#ff4d4f' }} />;
      case 'save':
        return <SaveOutlined style={{ color: '#722ed1' }} />;
      case 'rollback':
        return <HistoryOutlined style={{ color: '#fa8c16' }} />;
      default:
        return <ClockCircleOutlined />;
    }
  };

  const getChangeTypeColor = (type: string) => {
    switch (type) {
      case 'parameter_change':
        return 'blue';
      case 'deployment':
        return 'green';
      case 'pause':
        return 'orange';
      case 'stop':
        return 'red';
      case 'save':
        return 'purple';
      case 'rollback':
        return 'volcano';
      default:
        return 'default';
    }
  };

  const formatChangeDescription = (change: ConfigurationChange) => {
    switch (change.change_type) {
      case 'parameter_change':
        return `Updated ${change.changed_fields?.length || 0} parameter(s): ${change.changed_fields?.join(', ') || 'unknown'}`;
      case 'deployment':
        return `Strategy deployed to ${change.deployment_mode || 'unknown'} mode`;
      case 'pause':
        return 'Strategy execution paused';
      case 'stop':
        return 'Strategy execution stopped';
      case 'save':
        return 'Configuration saved as draft';
      case 'rollback':
        return `Rolled back to version ${change.rollback_version || 'unknown'}`;
      default:
        return change.description || 'Configuration changed';
    }
  };

  const calculatePerformanceImpact = (change: ConfigurationChange) => {
    if (!performanceHistory.length) return null;

    const changeTime = new Date(change.timestamp);
    const beforeMetrics = performanceHistory
      .filter(m => new Date(m.timestamp) < changeTime)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 5);

    const afterMetrics = performanceHistory
      .filter(m => new Date(m.timestamp) >= changeTime)
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .slice(0, 5);

    if (!beforeMetrics.length || !afterMetrics.length) return null;

    const avgBefore = beforeMetrics.reduce((sum, m) => sum + m.total_pnl.toNumber(), 0) / beforeMetrics.length;
    const avgAfter = afterMetrics.reduce((sum, m) => sum + m.total_pnl.toNumber(), 0) / afterMetrics.length;

    const impact = avgAfter - avgBefore;
    return {
      impact,
      percentage: avgBefore !== 0 ? ((impact / Math.abs(avgBefore)) * 100) : 0,
      direction: impact > 0 ? 'positive' : impact < 0 ? 'negative' : 'neutral'
    };
  };

  const renderTimelineItem = (change: ConfigurationChange) => {
    const performanceImpact = calculatePerformanceImpact(change);

    return {
      dot: getChangeTypeIcon(change.change_type),
      children: (
        <div>
          <Space direction="vertical" size="small" style={{ width: '100%' }}>
            <Row justify="space-between" align="middle">
              <Col>
                <Space>
                  <Text strong>{formatChangeDescription(change)}</Text>
                  <Tag color={getChangeTypeColor(change.change_type)}>
                    {change.change_type.replace('_', ' ').toUpperCase()}
                  </Tag>
                  {change.auto_generated && (
                    <Tag color="cyan">AUTO</Tag>
                  )}
                </Space>
              </Col>
              <Col>
                <Space>
                  {performanceImpact && (
                    <Tooltip title={`Performance impact: ${performanceImpact.percentage.toFixed(1)}%`}>
                      <Badge
                        count={performanceImpact.direction === 'positive' ? '↑' : performanceImpact.direction === 'negative' ? '↓' : '→'}
                        style={{
                          backgroundColor: performanceImpact.direction === 'positive' ? '#52c41a' : 
                                         performanceImpact.direction === 'negative' ? '#ff4d4f' : '#d9d9d9'
                        }}
                      />
                    </Tooltip>
                  )}
                  <Button
                    icon={<EyeOutlined />}
                    size="small"
                    onClick={() => setSelectedChange(change)}
                  >
                    Details
                  </Button>
                </Space>
              </Col>
            </Row>

            <Space split={<Divider type="vertical" />}>
              <Text type="secondary">
                {new Date(change.timestamp).toLocaleString()}
              </Text>
              <Text type="secondary">
                by {change.changed_by}
              </Text>
              {change.version && (
                <Text type="secondary">
                  Version {change.version}
                </Text>
              )}
            </Space>

            {change.reason && (
              <Paragraph type="secondary" style={{ margin: 0 }}>
                Reason: {change.reason}
              </Paragraph>
            )}

            {performanceImpact && (
              <Space>
                <Text type="secondary">Performance Impact:</Text>
                <Text
                  type={performanceImpact.direction === 'positive' ? 'success' : 
                        performanceImpact.direction === 'negative' ? 'danger' : 'secondary'}
                >
                  ${performanceImpact.impact.toFixed(2)} ({performanceImpact.percentage.toFixed(1)}%)
                </Text>
              </Space>
            )}
          </Space>
        </div>
      )
    };
  };

  const renderChangeDetails = () => {
    if (!selectedChange) return null;

    return (
      <Modal
        title={`Configuration Change Details - ${selectedChange.change_type.replace('_', ' ').toUpperCase()}`}
        open={!!selectedChange}
        onCancel={() => setSelectedChange(null)}
        width={900}
        footer={[
          selectedChange.config_snapshot && onRestoreConfig && (
            <Button
              key="restore"
              type="primary"
              onClick={() => {
                onRestoreConfig(selectedChange.config_snapshot!);
                setSelectedChange(null);
              }}
            >
              Restore This Configuration
            </Button>
          ),
          <Button key="close" onClick={() => setSelectedChange(null)}>
            Close
          </Button>
        ].filter(Boolean)}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Descriptions bordered column={2} size="small">
            <Descriptions.Item label="Change Type" span={1}>
              <Tag color={getChangeTypeColor(selectedChange.change_type)}>
                {selectedChange.change_type.replace('_', ' ').toUpperCase()}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Timestamp" span={1}>
              {new Date(selectedChange.timestamp).toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label="Changed By" span={1}>
              {selectedChange.changed_by}
            </Descriptions.Item>
            <Descriptions.Item label="Version" span={1}>
              {selectedChange.version || 'N/A'}
            </Descriptions.Item>
            <Descriptions.Item label="Auto Generated" span={1}>
              {selectedChange.auto_generated ? 'Yes' : 'No'}
            </Descriptions.Item>
            <Descriptions.Item label="Deployment Mode" span={1}>
              {selectedChange.deployment_mode || 'N/A'}
            </Descriptions.Item>
            {selectedChange.reason && (
              <Descriptions.Item label="Reason" span={2}>
                {selectedChange.reason}
              </Descriptions.Item>
            )}
          </Descriptions>

          {selectedChange.changed_fields && selectedChange.changed_fields.length > 0 && (
            <>
              <Divider orientation="left">Changed Fields</Divider>
              <List
                size="small"
                dataSource={selectedChange.changed_fields}
                renderItem={field => <List.Item>{field}</List.Item>}
              />
            </>
          )}

          {selectedChange.config_diff && (
            <>
              <Divider orientation="left">Configuration Diff</Divider>
              <Card size="small">
                <pre style={{ maxHeight: 300, overflow: 'auto' }}>
                  {JSON.stringify(selectedChange.config_diff, null, 2)}
                </pre>
              </Card>
            </>
          )}

          {selectedChange.config_snapshot && (
            <>
              <Divider orientation="left">Full Configuration</Divider>
              <Card size="small">
                <pre style={{ maxHeight: 300, overflow: 'auto' }}>
                  {JSON.stringify(selectedChange.config_snapshot.config_data, null, 2)}
                </pre>
              </Card>
            </>
          )}

          {selectedChange.performance_before && selectedChange.performance_after && (
            <>
              <Divider orientation="left">Performance Impact</Divider>
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="Before Change"
                    value={selectedChange.performance_before.total_pnl.toNumber()}
                    precision={2}
                    prefix="$"
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="After Change"
                    value={selectedChange.performance_after.total_pnl.toNumber()}
                    precision={2}
                    prefix="$"
                  />
                </Col>
              </Row>
            </>
          )}
        </Space>
      </Modal>
    );
  };

  return (
    <>
      <Modal
        title={
          <Space>
            <HistoryOutlined />
            Configuration History
          </Space>
        }
        open={visible}
        onCancel={onClose}
        width={1200}
        footer={[
          <Button key="close" onClick={onClose}>
            Close
          </Button>
        ]}
      >
        <Spin spinning={loading}>
          {history.length > 0 ? (
            <Timeline
              mode="left"
              items={history.map(renderTimelineItem)}
            />
          ) : (
            <Alert
              message="No Configuration History"
              description="This strategy has no configuration changes recorded yet."
              type="info"
              showIcon
            />
          )}
        </Spin>

        {auditLog.length > 0 && (
          <>
            <Divider orientation="left">Audit Log</Divider>
            <List
              size="small"
              dataSource={auditLog}
              renderItem={entry => (
                <List.Item>
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    <Row justify="space-between">
                      <Col>
                        <Space>
                          <Text strong>{entry.action}</Text>
                          {entry.risk_level && (
                            <Tag color={entry.risk_level === 'high' ? 'red' : entry.risk_level === 'medium' ? 'orange' : 'green'}>
                              {entry.risk_level.toUpperCase()} RISK
                            </Tag>
                          )}
                        </Space>
                      </Col>
                      <Col>
                        <Text type="secondary">
                          {new Date(entry.timestamp).toLocaleString()}
                        </Text>
                      </Col>
                    </Row>
                    {entry.details && (
                      <Text type="secondary">{entry.details}</Text>
                    )}
                    {entry.warnings && entry.warnings.length > 0 && (
                      <Alert
                        message="Warnings"
                        description={
                          <ul>
                            {entry.warnings.map((warning, idx) => (
                              <li key={idx}>{warning}</li>
                            ))}
                          </ul>
                        }
                        type="warning"
                        showIcon
                        size="small"
                      />
                    )}
                  </Space>
                </List.Item>
              )}
              pagination={{
                pageSize: 10,
                size: 'small'
              }}
            />
          </>
        )}
      </Modal>

      {renderChangeDetails()}
    </>
  );
};

export default ConfigurationHistory;