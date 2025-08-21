import React, { useState, useEffect } from 'react';
import {
  Card,
  Steps,
  Button,
  Progress,
  Statistic,
  Alert,
  Space,
  Typography,
  Row,
  Col,
  Table,
  Tag,
  Modal,
  Form,
  InputNumber,
  Select,
  Descriptions,
  Timeline,
  Tooltip,
  message
} from 'antd';
import {
  RocketOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  WarningOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  SettingOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined
} from '@ant-design/icons';
import type {
  RolloutPlan,
  RolloutPhase,
  SuccessCriteria,
  LiveStrategy,
  StrategyDeployment
} from '../../types/deployment';

const { Step } = Steps;
const { Text, Title } = Typography;
const { Option } = Select;

interface GradualRolloutControlsProps {
  deployment: StrategyDeployment;
  liveStrategy?: LiveStrategy;
  onPhaseAdvance?: (phaseIndex: number) => void;
  onRolloutModification?: (newPlan: RolloutPlan) => void;
  onRolloutPause?: () => void;
  onRolloutStop?: () => void;
}

interface RolloutProgress {
  currentPhase: number;
  phaseStartTime?: Date;
  phaseElapsedTime: number;
  phaseDuration: number;
  phaseProgress: number;
  criteriaStatus: { [key: string]: boolean };
  canAdvance: boolean;
  autoAdvance: boolean;
}

const GradualRolloutControls: React.FC<GradualRolloutControlsProps> = ({
  deployment,
  liveStrategy,
  onPhaseAdvance,
  onRolloutModification,
  onRolloutPause,
  onRolloutStop
}) => {
  const [rolloutProgress, setRolloutProgress] = useState<RolloutProgress>({
    currentPhase: deployment.rolloutPlan.currentPhase,
    phaseElapsedTime: 0,
    phaseDuration: 0,
    phaseProgress: 0,
    criteriaStatus: {},
    canAdvance: false,
    autoAdvance: true
  });
  
  const [modifyModalVisible, setModifyModalVisible] = useState(false);
  const [phaseDetailsVisible, setPhaseDetailsVisible] = useState(false);
  const [selectedPhase, setSelectedPhase] = useState<RolloutPhase | null>(null);
  const [rolloutForm] = Form.useForm();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    updateRolloutProgress();
    const interval = setInterval(updateRolloutProgress, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, [deployment, liveStrategy]);

  const updateRolloutProgress = () => {
    const currentPhase = rolloutProgress.currentPhase;
    const phase = deployment.rolloutPlan.phases[currentPhase];
    
    if (!phase) return;

    // Calculate phase progress
    const phaseStartTime = rolloutProgress.phaseStartTime || new Date();
    const elapsedTime = Date.now() - phaseStartTime.getTime();
    const progress = phase.duration > 0 ? Math.min((elapsedTime / (phase.duration * 1000)) * 100, 100) : 0;

    // Check success criteria
    const criteriaStatus: { [key: string]: boolean } = {};
    if (liveStrategy) {
      criteriaStatus.trades = !phase.successCriteria.minTrades || 
        liveStrategy.performanceMetrics.total_trades >= phase.successCriteria.minTrades;
      
      criteriaStatus.drawdown = !phase.successCriteria.maxDrawdown || 
        liveStrategy.riskMetrics.currentDrawdown <= (phase.successCriteria.maxDrawdown * 100);
      
      criteriaStatus.pnl = !phase.successCriteria.pnlThreshold || 
        (liveStrategy.realizedPnL + liveStrategy.unrealizedPnL) >= (phase.successCriteria.pnlThreshold * 1000);
    }

    const allCriteriaMet = Object.values(criteriaStatus).every(Boolean);
    const canAdvance = allCriteriaMet && (phase.duration === -1 || progress >= 100);

    setRolloutProgress(prev => ({
      ...prev,
      phaseElapsedTime: elapsedTime,
      phaseDuration: phase.duration * 1000,
      phaseProgress: progress,
      criteriaStatus,
      canAdvance
    }));

    // Auto-advance if enabled and criteria met
    if (rolloutProgress.autoAdvance && canAdvance && currentPhase < deployment.rolloutPlan.phases.length - 1) {
      advancePhase();
    }
  };

  const advancePhase = async () => {
    const nextPhase = rolloutProgress.currentPhase + 1;
    if (nextPhase >= deployment.rolloutPlan.phases.length) {
      message.success('Rollout completed successfully!');
      return;
    }

    setLoading(true);
    try {
      await onPhaseAdvance?.(nextPhase);
      setRolloutProgress(prev => ({
        ...prev,
        currentPhase: nextPhase,
        phaseStartTime: new Date(),
        phaseElapsedTime: 0,
        phaseProgress: 0,
        criteriaStatus: {},
        canAdvance: false
      }));
      message.success(`Advanced to ${deployment.rolloutPlan.phases[nextPhase].name}`);
    } catch (error) {
      console.error('Failed to advance phase:', error);
      message.error('Failed to advance to next phase');
    } finally {
      setLoading(false);
    }
  };

  const modifyRolloutPlan = async (values: any) => {
    setLoading(true);
    try {
      const newPlan: RolloutPlan = {
        ...deployment.rolloutPlan,
        phases: values.phases
      };
      
      await onRolloutModification?.(newPlan);
      setModifyModalVisible(false);
      message.success('Rollout plan updated successfully');
    } catch (error) {
      console.error('Failed to modify rollout plan:', error);
      message.error('Failed to update rollout plan');
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (milliseconds: number): string => {
    if (milliseconds < 0) return 'Indefinite';
    const minutes = Math.floor(milliseconds / 60000);
    const hours = Math.floor(minutes / 60);
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    }
    return `${minutes}m`;
  };

  const getCurrentPhase = (): RolloutPhase | null => {
    return deployment.rolloutPlan.phases[rolloutProgress.currentPhase] || null;
  };

  const renderPhaseSteps = () => {
    const currentPhase = rolloutProgress.currentPhase;
    
    return (
      <Steps current={currentPhase} size="small" className="mb-6">
        {deployment.rolloutPlan.phases.map((phase, index) => (
          <Step
            key={index}
            title={phase.name.replace('_', ' ').toUpperCase()}
            description={`${phase.positionSizePercent}%`}
            icon={
              index < currentPhase ? <CheckCircleOutlined /> :
              index === currentPhase ? <ClockCircleOutlined /> :
              <PlayCircleOutlined />
            }
            status={
              index < currentPhase ? 'finish' :
              index === currentPhase ? 'process' : 'wait'
            }
          />
        ))}
      </Steps>
    );
  };

  const renderCurrentPhaseStatus = () => {
    const phase = getCurrentPhase();
    if (!phase) return null;

    return (
      <Card title="Current Phase Status" size="small" className="mb-4">
        <Row gutter={16} className="mb-4">
          <Col span={6}>
            <Statistic
              title="Phase"
              value={phase.name.replace('_', ' ').toUpperCase()}
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Position Size"
              value={phase.positionSizePercent}
              suffix="%"
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Duration"
              value={formatDuration(rolloutProgress.phaseDuration)}
              valueStyle={{ fontSize: '16px' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Progress"
              value={rolloutProgress.phaseProgress}
              precision={1}
              suffix="%"
              valueStyle={{ color: rolloutProgress.phaseProgress >= 100 ? '#52c41a' : '#1890ff' }}
            />
          </Col>
        </Row>

        {phase.duration > 0 && (
          <div className="mb-4">
            <Text strong>Time Progress:</Text>
            <Progress
              percent={rolloutProgress.phaseProgress}
              status={rolloutProgress.phaseProgress >= 100 ? 'success' : 'active'}
              strokeColor="#1890ff"
            />
          </div>
        )}

        <div className="mb-4">
          <Text strong>Success Criteria:</Text>
          <div className="grid grid-cols-2 gap-2 mt-2">
            {phase.successCriteria.minTrades && (
              <div className="flex items-center space-x-2">
                {rolloutProgress.criteriaStatus.trades ? (
                  <CheckCircleOutlined style={{ color: '#52c41a' }} />
                ) : (
                  <ClockCircleOutlined style={{ color: '#faad14' }} />
                )}
                <Text>Min Trades: {liveStrategy?.performanceMetrics.total_trades || 0}/{phase.successCriteria.minTrades}</Text>
              </div>
            )}
            
            {phase.successCriteria.maxDrawdown && (
              <div className="flex items-center space-x-2">
                {rolloutProgress.criteriaStatus.drawdown ? (
                  <CheckCircleOutlined style={{ color: '#52c41a' }} />
                ) : (
                  <WarningOutlined style={{ color: '#f5222d' }} />
                )}
                <Text>Max Drawdown: {liveStrategy?.riskMetrics.currentDrawdown.toFixed(2) || 0}%/{(phase.successCriteria.maxDrawdown * 100).toFixed(1)}%</Text>
              </div>
            )}
            
            {phase.successCriteria.pnlThreshold && (
              <div className="flex items-center space-x-2">
                {rolloutProgress.criteriaStatus.pnl ? (
                  <CheckCircleOutlined style={{ color: '#52c41a' }} />
                ) : (
                  <ArrowDownOutlined style={{ color: '#f5222d' }} />
                )}
                <Text>
                  P&L: ${((liveStrategy?.realizedPnL || 0) + (liveStrategy?.unrealizedPnL || 0)).toFixed(2)}/
                  ${(phase.successCriteria.pnlThreshold * 1000).toFixed(2)}
                </Text>
              </div>
            )}

            {phase.successCriteria.ongoing && (
              <div className="flex items-center space-x-2">
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
                <Text>Ongoing monitoring active</Text>
              </div>
            )}
          </div>
        </div>

        {rolloutProgress.canAdvance && (
          <Alert
            message="Phase Ready for Advancement"
            description="All success criteria have been met. The strategy can advance to the next phase."
            type="success"
            action={
              <Space>
                <Button
                  size="small"
                  type="primary"
                  onClick={advancePhase}
                  loading={loading}
                >
                  Advance Phase
                </Button>
              </Space>
            }
            className="mb-4"
          />
        )}
      </Card>
    );
  };

  const renderPhaseTable = () => {
    const columns = [
      {
        title: 'Phase',
        dataIndex: 'name',
        render: (name: string, _, index: number) => (
          <div>
            <Tag color={index === rolloutProgress.currentPhase ? 'blue' : index < rolloutProgress.currentPhase ? 'green' : 'default'}>
              {name.replace('_', ' ').toUpperCase()}
            </Tag>
            {index === rolloutProgress.currentPhase && <Text type="secondary"> (Current)</Text>}
          </div>
        )
      },
      {
        title: 'Position Size',
        dataIndex: 'positionSizePercent',
        render: (size: number) => `${size}%`
      },
      {
        title: 'Duration',
        dataIndex: 'duration',
        render: (duration: number) => formatDuration(duration * 1000)
      },
      {
        title: 'Success Criteria',
        key: 'criteria',
        render: (_, phase: RolloutPhase) => (
          <div>
            {phase.successCriteria.minTrades && <div>Min trades: {phase.successCriteria.minTrades}</div>}
            {phase.successCriteria.maxDrawdown && <div>Max DD: {(phase.successCriteria.maxDrawdown * 100).toFixed(1)}%</div>}
            {phase.successCriteria.pnlThreshold && <div>Min P&L: ${(phase.successCriteria.pnlThreshold * 1000).toFixed(0)}</div>}
            {phase.successCriteria.ongoing && <div>Ongoing monitoring</div>}
          </div>
        )
      },
      {
        title: 'Actions',
        key: 'actions',
        render: (_, phase: RolloutPhase, index: number) => (
          <Space size="small">
            <Button
              size="small"
              icon={<SettingOutlined />}
              onClick={() => {
                setSelectedPhase(phase);
                setPhaseDetailsVisible(true);
              }}
            >
              Details
            </Button>
            {index === rolloutProgress.currentPhase && rolloutProgress.canAdvance && (
              <Button
                size="small"
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={advancePhase}
                loading={loading}
              >
                Advance
              </Button>
            )}
          </Space>
        )
      }
    ];

    return (
      <Card title="Rollout Timeline" size="small">
        <Table
          dataSource={deployment.rolloutPlan.phases}
          columns={columns}
          pagination={false}
          size="small"
          rowKey="name"
        />
      </Card>
    );
  };

  return (
    <div className="gradual-rollout-controls">
      <Card
        title={
          <div className="flex items-center space-x-2">
            <RocketOutlined className="text-blue-600" />
            <span>Gradual Rollout Control</span>
          </div>
        }
        extra={
          <Space>
            <Button
              icon={<SettingOutlined />}
              onClick={() => {
                rolloutForm.setFieldsValue({ phases: deployment.rolloutPlan.phases });
                setModifyModalVisible(true);
              }}
            >
              Modify Plan
            </Button>
            <Button
              icon={<PauseCircleOutlined />}
              onClick={onRolloutPause}
            >
              Pause Rollout
            </Button>
            <Button
              danger
              icon={<StopOutlined />}
              onClick={onRolloutStop}
            >
              Stop Rollout
            </Button>
          </Space>
        }
      >
        {renderPhaseSteps()}
        {renderCurrentPhaseStatus()}
        {renderPhaseTable()}
      </Card>

      {/* Modify Rollout Plan Modal */}
      <Modal
        title="Modify Rollout Plan"
        visible={modifyModalVisible}
        onCancel={() => setModifyModalVisible(false)}
        footer={null}
        width={800}
      >
        <Alert
          message="Rollout Modification Warning"
          description="Modifying the rollout plan will affect the current deployment. Make sure the changes are safe."
          type="warning"
          className="mb-4"
        />

        <Form form={rolloutForm} layout="vertical" onFinish={modifyRolloutPlan}>
          <Form.List name="phases">
            {(fields, { add, remove }) => (
              <>
                {fields.map(({ key, name, ...restField }, index) => (
                  <Card key={key} title={`Phase ${index + 1}`} size="small" className="mb-4">
                    <Row gutter={16}>
                      <Col span={12}>
                        <Form.Item
                          {...restField}
                          name={[name, 'name']}
                          label="Phase Name"
                          required
                        >
                          <Input placeholder="e.g., validation" />
                        </Form.Item>
                      </Col>
                      <Col span={6}>
                        <Form.Item
                          {...restField}
                          name={[name, 'positionSizePercent']}
                          label="Position Size %"
                          required
                        >
                          <InputNumber min={1} max={100} />
                        </Form.Item>
                      </Col>
                      <Col span={6}>
                        <Form.Item
                          {...restField}
                          name={[name, 'duration']}
                          label="Duration (seconds)"
                          required
                        >
                          <InputNumber min={-1} placeholder="-1 for indefinite" />
                        </Form.Item>
                      </Col>
                    </Row>

                    <Row gutter={16}>
                      <Col span={8}>
                        <Form.Item
                          {...restField}
                          name={[name, 'successCriteria', 'minTrades']}
                          label="Min Trades"
                        >
                          <InputNumber min={0} />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item
                          {...restField}
                          name={[name, 'successCriteria', 'maxDrawdown']}
                          label="Max Drawdown"
                        >
                          <InputNumber min={0} max={1} step={0.01} />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item
                          {...restField}
                          name={[name, 'successCriteria', 'pnlThreshold']}
                          label="P&L Threshold"
                        >
                          <InputNumber step={0.01} />
                        </Form.Item>
                      </Col>
                    </Row>

                    <Button danger onClick={() => remove(name)}>
                      Remove Phase
                    </Button>
                  </Card>
                ))}
                
                <Button type="dashed" onClick={() => add()} block>
                  Add Phase
                </Button>
              </>
            )}
          </Form.List>

          <div className="flex justify-end space-x-2 mt-6">
            <Button onClick={() => setModifyModalVisible(false)}>
              Cancel
            </Button>
            <Button type="primary" htmlType="submit" loading={loading}>
              Update Rollout Plan
            </Button>
          </div>
        </Form>
      </Modal>

      {/* Phase Details Modal */}
      <Modal
        title="Phase Details"
        visible={phaseDetailsVisible}
        onCancel={() => setPhaseDetailsVisible(false)}
        footer={null}
      >
        {selectedPhase && (
          <Descriptions bordered>
            <Descriptions.Item label="Phase Name" span={3}>
              {selectedPhase.name.replace('_', ' ').toUpperCase()}
            </Descriptions.Item>
            <Descriptions.Item label="Position Size" span={3}>
              {selectedPhase.positionSizePercent}%
            </Descriptions.Item>
            <Descriptions.Item label="Duration" span={3}>
              {formatDuration(selectedPhase.duration * 1000)}
            </Descriptions.Item>
            <Descriptions.Item label="Min Trades" span={3}>
              {selectedPhase.successCriteria.minTrades || 'Not specified'}
            </Descriptions.Item>
            <Descriptions.Item label="Max Drawdown" span={3}>
              {selectedPhase.successCriteria.maxDrawdown ? 
                `${(selectedPhase.successCriteria.maxDrawdown * 100).toFixed(1)}%` : 
                'Not specified'}
            </Descriptions.Item>
            <Descriptions.Item label="P&L Threshold" span={3}>
              {selectedPhase.successCriteria.pnlThreshold ? 
                `$${(selectedPhase.successCriteria.pnlThreshold * 1000).toFixed(2)}` : 
                'Not specified'}
            </Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </div>
  );
};

export default GradualRolloutControls;