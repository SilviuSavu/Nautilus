import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Table,
  Button,
  Typography,
  Space,
  Tag,
  Alert,
  Modal,
  Form,
  Select,
  InputNumber,
  Switch,
  Progress,
  Tooltip,
  Badge,
  Divider,
  notification,
  Row,
  Col,
  Statistic,
  Descriptions
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  WarningOutlined,
  ExclamationCircleOutlined,
  SyncOutlined,
  ThunderboltOutlined,
  ClusterOutlined,
  BarChartOutlined,
  SettingOutlined,
  BugOutlined
} from '@ant-design/icons';

import { StrategyConfig, StrategyInstance, StrategyState } from './types/strategyTypes';
import strategyService from './services/strategyService';

const { Title, Text } = Typography;
const { Option } = Select;

interface ConflictDetection {
  type: 'instrument_overlap' | 'risk_limit' | 'correlation' | 'resource_contention';
  severity: 'warning' | 'error' | 'critical';
  strategies: string[];
  message: string;
  recommendation: string;
}

interface CoordinationRule {
  id: string;
  name: string;
  type: 'priority' | 'resource_limit' | 'risk_limit' | 'position_sizing';
  enabled: boolean;
  parameters: Record<string, any>;
}

interface MultiStrategyCoordinatorProps {
  strategies: StrategyConfig[];
  instances: Record<string, StrategyInstance>;
  onInstanceUpdate?: (strategyId: string, instance: StrategyInstance) => void;
  className?: string;
}

export const MultiStrategyCoordinator: React.FC<MultiStrategyCoordinatorProps> = ({
  strategies,
  instances,
  onInstanceUpdate,
  className
}) => {
  const [conflicts, setConflicts] = useState<ConflictDetection[]>([]);
  const [coordinationRules, setCoordinationRules] = useState<CoordinationRule[]>([]);
  const [loading, setLoading] = useState(false);
  const [rulesModalVisible, setRulesModalVisible] = useState(false);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [batchAction, setBatchAction] = useState<'start' | 'stop' | 'pause' | null>(null);
  const [form] = Form.useForm();

  // Initialize default coordination rules
  useEffect(() => {
    setCoordinationRules([
      {
        id: 'priority-1',
        name: 'High Priority Strategies First',
        type: 'priority',
        enabled: true,
        parameters: { order: 'priority_desc' }
      },
      {
        id: 'risk-1',
        name: 'Total Portfolio Risk Limit',
        type: 'risk_limit',
        enabled: true,
        parameters: { max_portfolio_risk: 0.02 }
      },
      {
        id: 'resource-1',
        name: 'Maximum Concurrent Strategies',
        type: 'resource_limit',
        enabled: true,
        parameters: { max_concurrent: 5 }
      },
      {
        id: 'position-1',
        name: 'Instrument Position Limit',
        type: 'position_sizing',
        enabled: true,
        parameters: { max_per_instrument: 0.1 }
      }
    ]);
  }, []);

  // Detect conflicts whenever strategies or instances change
  useEffect(() => {
    detectConflicts();
  }, [strategies, instances]);

  const detectConflicts = useCallback(() => {
    const detectedConflicts: ConflictDetection[] = [];
    const runningInstances = Object.values(instances).filter(i => i.state === 'running');

    // Instrument overlap detection
    const instrumentMap: Record<string, string[]> = {};
    runningInstances.forEach(instance => {
      const strategy = strategies.find(s => s.id === instance.config_id);
      if (strategy) {
        const instruments = extractInstruments(strategy);
        instruments.forEach(instrument => {
          if (!instrumentMap[instrument]) {
            instrumentMap[instrument] = [];
          }
          instrumentMap[instrument].push(strategy.name);
        });
      }
    });

    Object.entries(instrumentMap).forEach(([instrument, strategyNames]) => {
      if (strategyNames.length > 1) {
        detectedConflicts.push({
          type: 'instrument_overlap',
          severity: 'warning',
          strategies: strategyNames,
          message: `Multiple strategies trading ${instrument}`,
          recommendation: 'Consider position sizing coordination or strategy prioritization'
        });
      }
    });

    // Risk limit detection
    const totalRisk = runningInstances.reduce((total, instance) => {
      const strategy = strategies.find(s => s.id === instance.config_id);
      return total + (strategy?.risk_settings?.max_position_size || 0);
    }, 0);

    if (totalRisk > 10000) { // Example limit
      detectedConflicts.push({
        type: 'risk_limit',
        severity: 'error',
        strategies: runningInstances.map(i => {
          const strategy = strategies.find(s => s.id === i.config_id);
          return strategy?.name || 'Unknown';
        }),
        message: `Total portfolio risk exceeds limit: $${totalRisk.toFixed(2)}`,
        recommendation: 'Reduce position sizes or stop some strategies'
      });
    }

    // Resource contention detection
    if (runningInstances.length > 5) {
      detectedConflicts.push({
        type: 'resource_contention',
        severity: 'warning',
        strategies: runningInstances.map(i => {
          const strategy = strategies.find(s => s.id === i.config_id);
          return strategy?.name || 'Unknown';
        }),
        message: `High number of concurrent strategies: ${runningInstances.length}`,
        recommendation: 'Monitor system performance and consider strategy consolidation'
      });
    }

    setConflicts(detectedConflicts);
  }, [strategies, instances]);

  const extractInstruments = (strategy: StrategyConfig): string[] => {
    // Extract instruments from strategy parameters
    const instruments: string[] = [];
    if (strategy.parameters.instrument_id) {
      instruments.push(strategy.parameters.instrument_id);
    }
    if (strategy.parameters.instruments) {
      instruments.push(...strategy.parameters.instruments);
    }
    return instruments;
  };

  const handleBatchAction = async () => {
    if (!batchAction || selectedStrategies.length === 0) return;

    try {
      setLoading(true);
      const promises = selectedStrategies.map(async (strategyId) => {
        const instance = instances[strategyId];
        if (instance) {
          return strategyService.controlStrategy(instance.id, { action: batchAction });
        }
      });

      await Promise.all(promises.filter(Boolean));

      notification.success({
        message: 'Batch Action Completed',
        description: `${batchAction} action applied to ${selectedStrategies.length} strategies`,
        duration: 3
      });

      setSelectedStrategies([]);
      setBatchAction(null);

    } catch (error: any) {
      notification.error({
        message: 'Batch Action Failed',
        description: error.message || 'Failed to execute batch action',
        duration: 4
      });
    } finally {
      setLoading(false);
    }
  };

  const getConflictIcon = (severity: ConflictDetection['severity']) => {
    switch (severity) {
      case 'critical': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'error': return <ExclamationCircleOutlined style={{ color: '#ff7a45' }} />;
      case 'warning': return <WarningOutlined style={{ color: '#fa8c16' }} />;
      default: return <WarningOutlined />;
    }
  };

  const getStateColor = (state: StrategyState): string => {
    switch (state) {
      case 'running': return 'success';
      case 'paused': return 'warning';
      case 'stopped': return 'default';
      case 'error': return 'error';
      case 'initializing': return 'processing';
      default: return 'default';
    }
  };

  const getTotalMetrics = () => {
    const runningInstances = Object.values(instances).filter(i => i.state === 'running');
    const totalPnL = runningInstances.reduce((total, instance) => {
      return total + Number(instance.performance_metrics?.total_pnl || 0);
    }, 0);
    
    const totalTrades = runningInstances.reduce((total, instance) => {
      return total + (instance.performance_metrics?.total_trades || 0);
    }, 0);

    const avgWinRate = runningInstances.length > 0 
      ? runningInstances.reduce((total, instance) => {
          return total + (instance.performance_metrics?.win_rate || 0);
        }, 0) / runningInstances.length
      : 0;

    return { totalPnL, totalTrades, avgWinRate, runningCount: runningInstances.length };
  };

  const columns = [
    {
      title: 'Strategy',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, strategy: StrategyConfig) => {
        const instance = instances[strategy.id];
        const hasConflicts = conflicts.some(c => c.strategies.includes(name));
        
        return (
          <div>
            <Space>
              <Text strong>{name}</Text>
              {hasConflicts && (
                <Tooltip title="Has conflicts">
                  <WarningOutlined style={{ color: '#fa8c16' }} />
                </Tooltip>
              )}
            </Space>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {strategy.template_id}
            </Text>
          </div>
        );
      }
    },
    {
      title: 'Status',
      key: 'status',
      render: (_, strategy: StrategyConfig) => {
        const instance = instances[strategy.id];
        if (!instance) {
          return <Tag color="default">Not Deployed</Tag>;
        }
        return <Tag color={getStateColor(instance.state)}>{instance.state.toUpperCase()}</Tag>;
      }
    },
    {
      title: 'P&L',
      key: 'pnl',
      render: (_, strategy: StrategyConfig) => {
        const instance = instances[strategy.id];
        if (!instance?.performance_metrics) {
          return <Text type="secondary">-</Text>;
        }
        const pnl = Number(instance.performance_metrics.total_pnl);
        return (
          <Text style={{ color: pnl >= 0 ? '#3f8600' : '#cf1322' }}>
            ${pnl.toFixed(2)}
          </Text>
        );
      }
    },
    {
      title: 'Trades',
      key: 'trades',
      render: (_, strategy: StrategyConfig) => {
        const instance = instances[strategy.id];
        return instance?.performance_metrics?.total_trades || 0;
      }
    },
    {
      title: 'Win Rate',
      key: 'win_rate',
      render: (_, strategy: StrategyConfig) => {
        const instance = instances[strategy.id];
        if (!instance?.performance_metrics) {
          return <Text type="secondary">-</Text>;
        }
        return `${(instance.performance_metrics.win_rate * 100).toFixed(1)}%`;
      }
    }
  ];

  const metrics = getTotalMetrics();

  return (
    <div className={`multi-strategy-coordinator ${className || ''}`}>
      <Card>
        <div style={{ marginBottom: 24 }}>
          <Title level={4}>
            <ClusterOutlined /> Multi-Strategy Coordination
          </Title>
          <Text type="secondary">
            Coordinate multiple strategies, detect conflicts, and manage portfolio risk
          </Text>
        </div>

        {/* Portfolio Overview */}
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={6}>
            <Card size="small">
              <Statistic
                title="Running Strategies"
                value={metrics.runningCount}
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card size="small">
              <Statistic
                title="Total P&L"
                value={metrics.totalPnL}
                precision={2}
                prefix="$"
                valueStyle={{ color: metrics.totalPnL >= 0 ? '#3f8600' : '#cf1322' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card size="small">
              <Statistic
                title="Total Trades"
                value={metrics.totalTrades}
                prefix={<BarChartOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={6}>
            <Card size="small">
              <Statistic
                title="Avg Win Rate"
                value={metrics.avgWinRate * 100}
                precision={1}
                suffix="%"
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
        </Row>

        {/* Conflict Alerts */}
        {conflicts.length > 0 && (
          <Alert
            type="warning"
            message={`${conflicts.length} Conflict${conflicts.length > 1 ? 's' : ''} Detected`}
            description={
              <div style={{ marginTop: 8 }}>
                {conflicts.map((conflict, index) => (
                  <div key={index} style={{ marginBottom: 8 }}>
                    <Space>
                      {getConflictIcon(conflict.severity)}
                      <Text strong>{conflict.message}</Text>
                    </Space>
                    <br />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      Affected: {conflict.strategies.join(', ')}
                    </Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      Recommendation: {conflict.recommendation}
                    </Text>
                  </div>
                ))}
              </div>
            }
            style={{ marginBottom: 16 }}
            showIcon
          />
        )}

        {/* Batch Controls */}
        <Card size="small" style={{ marginBottom: 16 }}>
          <Row gutter={[16, 16]} align="middle">
            <Col xs={24} md={12}>
              <Space>
                <Text strong>Batch Actions:</Text>
                <Select
                  placeholder="Select action"
                  value={batchAction}
                  onChange={setBatchAction}
                  style={{ width: 120 }}
                >
                  <Option value="start">Start</Option>
                  <Option value="pause">Pause</Option>
                  <Option value="stop">Stop</Option>
                </Select>
                <Button
                  type="primary"
                  loading={loading}
                  disabled={!batchAction || selectedStrategies.length === 0}
                  onClick={handleBatchAction}
                >
                  Execute ({selectedStrategies.length})
                </Button>
              </Space>
            </Col>
            <Col xs={24} md={12}>
              <div style={{ textAlign: 'right' }}>
                <Space>
                  <Button
                    icon={<SettingOutlined />}
                    onClick={() => setRulesModalVisible(true)}
                  >
                    Coordination Rules
                  </Button>
                  <Button
                    icon={<SyncOutlined />}
                    onClick={detectConflicts}
                  >
                    Refresh
                  </Button>
                </Space>
              </div>
            </Col>
          </Row>
        </Card>

        {/* Strategy Table */}
        <Table
          columns={columns}
          dataSource={strategies}
          rowKey="id"
          rowSelection={{
            selectedRowKeys: selectedStrategies,
            onChange: setSelectedStrategies,
            getCheckboxProps: (strategy) => ({
              disabled: !instances[strategy.id] // Only allow selection of deployed strategies
            })
          }}
          size="small"
          locale={{
            emptyText: 'No strategies available for coordination'
          }}
        />

        {/* Coordination Rules Modal */}
        <Modal
          title="Coordination Rules"
          open={rulesModalVisible}
          onCancel={() => setRulesModalVisible(false)}
          width={800}
          footer={[
            <Button key="cancel" onClick={() => setRulesModalVisible(false)}>
              Close
            </Button>,
            <Button
              key="save"
              type="primary"
              onClick={() => {
                notification.success({
                  message: 'Rules Updated',
                  description: 'Coordination rules have been updated successfully.',
                  duration: 3
                });
                setRulesModalVisible(false);
              }}
            >
              Save Rules
            </Button>
          ]}
        >
          <div>
            <Text type="secondary">
              Configure rules for coordinating multiple strategies
            </Text>
            <Divider />
            
            {coordinationRules.map(rule => (
              <Card key={rule.id} size="small" style={{ marginBottom: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <Text strong>{rule.name}</Text>
                    <br />
                    <Tag color="blue">{rule.type.replace('_', ' ').toUpperCase()}</Tag>
                  </div>
                  <Switch
                    checked={rule.enabled}
                    onChange={(checked) => {
                      setCoordinationRules(prev => 
                        prev.map(r => r.id === rule.id ? { ...r, enabled: checked } : r)
                      );
                    }}
                  />
                </div>
                
                {rule.enabled && (
                  <div style={{ marginTop: 12 }}>
                    <Descriptions size="small" column={1}>
                      {Object.entries(rule.parameters).map(([key, value]) => (
                        <Descriptions.Item
                          key={key}
                          label={key.replace('_', ' ').toUpperCase()}
                        >
                          {String(value)}
                        </Descriptions.Item>
                      ))}
                    </Descriptions>
                  </div>
                )}
              </Card>
            ))}
          </div>
        </Modal>
      </Card>
    </div>
  );
};