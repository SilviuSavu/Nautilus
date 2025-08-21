import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Form,
  Select,
  InputNumber,
  Table,
  Progress,
  Statistic,
  Tag,
  Space,
  Modal,
  Typography,
  Alert,
  Steps,
  message,
  Spin,
  Badge,
  Tooltip,
  Switch,
  Slider,
  Input
} from 'antd'
import {
  RocketOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  StopOutlined,
  PlayCircleOutlined,
  PauseOutlined,
  RollbackOutlined,
  SafetyOutlined,
  UserOutlined,
  ClockCircleOutlined,
  WarningOutlined
} from '@ant-design/icons'
import dayjs from 'dayjs'

const { Title, Text, Paragraph } = Typography
const { Step } = Steps
const { TextArea } = Input

interface DeploymentRequest {
  deploymentId?: string
  strategyId: string
  version: string
  backtestId: string
  proposedConfig: StrategyConfig
  riskAssessment: RiskAssessment
  rolloutPlan: RolloutPlan
  approvalRequired: boolean
  status?: 'draft' | 'pending_approval' | 'approved' | 'deploying' | 'deployed' | 'failed' | 'rejected'
}

interface StrategyConfig {
  positionSize: number
  maxDrawdown: number
  stopLoss: number
  takeProfit: number
  riskPerTrade: number
  instruments: string[]
  venues: string[]
}

interface RiskAssessment {
  riskScore: number
  portfolioImpact: number
  correlationRisk: number
  maxPositionSize: number
  recommendedSize: number
  warnings: string[]
}

interface RolloutPlan {
  phases: RolloutPhase[]
  currentPhase: number
  escalationCriteria: EscalationCriteria
}

interface RolloutPhase {
  name: string
  positionSizePercent: number
  duration: number
  successCriteria: SuccessCriteria
  rollbackTriggers: RollbackTrigger[]
}

interface EscalationCriteria {
  minSuccessRate: number
  maxDrawdown: number
  minProfitability: number
  evaluationPeriod: number
}

interface SuccessCriteria {
  minReturn: number
  maxDrawdown: number
  minWinRate: number
}

interface RollbackTrigger {
  metric: string
  threshold: number
  action: 'pause' | 'rollback' | 'reduce_size'
}

interface LiveStrategy {
  strategyInstanceId: string
  deploymentId: string
  strategyId: string
  version: string
  state: 'deploying' | 'running' | 'paused' | 'stopped' | 'error'
  currentPhase: number
  positionSize: number
  realizedPnL: number
  unrealizedPnL: number
  lastHeartbeat: string
  performanceMetrics: PerformanceMetrics
  riskMetrics: RiskMetrics
}

interface PerformanceMetrics {
  totalReturn: number
  dailyReturn: number
  winRate: number
  sharpeRatio: number
  maxDrawdown: number
  tradesCount: number
}

interface RiskMetrics {
  currentExposure: number
  portfolioWeight: number
  correlation: number
  var95: number
  beta: number
}

interface ApprovalWorkflow {
  approvalId: string
  deploymentId: string
  approver: string
  level: number
  status: 'pending' | 'approved' | 'rejected'
  comments?: string
  timestamp?: string
}

const StrategyDeploymentPipeline: React.FC = () => {
  const [deployments, setDeployments] = useState<DeploymentRequest[]>([])
  const [liveStrategies, setLiveStrategies] = useState<LiveStrategy[]>([])
  const [selectedDeployment, setSelectedDeployment] = useState<DeploymentRequest | null>(null)
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [approvalModalVisible, setApprovalModalVisible] = useState(false)
  const [emergencyModalVisible, setEmergencyModalVisible] = useState(false)
  const [loading, setLoading] = useState(false)
  const [form] = Form.useForm()

  const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002'

  useEffect(() => {
    loadDeployments()
    loadLiveStrategies()

    const interval = setInterval(() => {
      loadLiveStrategies()
    }, 10000) // Update live strategies every 10 seconds

    return () => clearInterval(interval)
  }, [])

  const loadDeployments = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/nautilus/deployment/list`)
      if (response.ok) {
        const data = await response.json()
        setDeployments(data.deployments || [])
      }
    } catch (error) {
      console.error('Failed to load deployments:', error)
    }
  }

  const loadLiveStrategies = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/nautilus/strategies/live`)
      if (response.ok) {
        const data = await response.json()
        setLiveStrategies(data.strategies || [])
      }
    } catch (error) {
      console.error('Failed to load live strategies:', error)
    }
  }

  const createDeploymentRequest = async (values: any) => {
    setLoading(true)
    try {
      const request: DeploymentRequest = {
        strategyId: values.strategyId,
        version: values.version,
        backtestId: values.backtestId,
        proposedConfig: {
          positionSize: values.positionSize,
          maxDrawdown: values.maxDrawdown,
          stopLoss: values.stopLoss,
          takeProfit: values.takeProfit,
          riskPerTrade: values.riskPerTrade,
          instruments: values.instruments,
          venues: values.venues
        },
        riskAssessment: {
          riskScore: 0.75,
          portfolioImpact: 0.15,
          correlationRisk: 0.3,
          maxPositionSize: 100000,
          recommendedSize: values.positionSize * 0.8,
          warnings: []
        },
        rolloutPlan: {
          phases: [
            {
              name: 'Phase 1: Limited Test',
              positionSizePercent: 25,
              duration: 24, // hours
              successCriteria: { minReturn: 0, maxDrawdown: 2, minWinRate: 40 },
              rollbackTriggers: [
                { metric: 'drawdown', threshold: 3, action: 'pause' },
                { metric: 'return', threshold: -5, action: 'rollback' }
              ]
            },
            {
              name: 'Phase 2: Gradual Scale',
              positionSizePercent: 50,
              duration: 48,
              successCriteria: { minReturn: 1, maxDrawdown: 3, minWinRate: 45 },
              rollbackTriggers: [
                { metric: 'drawdown', threshold: 5, action: 'reduce_size' },
                { metric: 'return', threshold: -7, action: 'rollback' }
              ]
            },
            {
              name: 'Phase 3: Full Deployment',
              positionSizePercent: 100,
              duration: 168, // 1 week
              successCriteria: { minReturn: 2, maxDrawdown: 5, minWinRate: 50 },
              rollbackTriggers: [
                { metric: 'drawdown', threshold: 8, action: 'pause' },
                { metric: 'return', threshold: -10, action: 'rollback' }
              ]
            }
          ],
          currentPhase: 0,
          escalationCriteria: {
            minSuccessRate: 60,
            maxDrawdown: 5,
            minProfitability: 1.5,
            evaluationPeriod: 24
          }
        },
        approvalRequired: values.positionSize > 50000 || values.riskPerTrade > 2
      }

      const response = await fetch(`${apiUrl}/api/v1/nautilus/deployment/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      })

      if (response.ok) {
        const result = await response.json()
        message.success(`Deployment request created: ${result.deploymentId}`)
        setCreateModalVisible(false)
        form.resetFields()
        loadDeployments()
      } else {
        const error = await response.json()
        message.error(`Failed to create deployment: ${error.detail}`)
      }
    } catch (error) {
      message.error('Failed to create deployment request')
      console.error('Create deployment error:', error)
    } finally {
      setLoading(false)
    }
  }

  const approveDeployment = async (deploymentId: string, comments: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/nautilus/deployment/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deploymentId, comments })
      })

      if (response.ok) {
        message.success('Deployment approved')
        setApprovalModalVisible(false)
        loadDeployments()
      } else {
        message.error('Failed to approve deployment')
      }
    } catch (error) {
      message.error('Failed to approve deployment')
      console.error('Approve deployment error:', error)
    }
  }

  const deployStrategy = async (deploymentId: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/nautilus/deployment/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deploymentId })
      })

      if (response.ok) {
        message.success('Strategy deployment started')
        loadDeployments()
        loadLiveStrategies()
      } else {
        message.error('Failed to deploy strategy')
      }
    } catch (error) {
      message.error('Failed to deploy strategy')
      console.error('Deploy strategy error:', error)
    }
  }

  const emergencyStop = async (strategyInstanceId: string, reason: string) => {
    Modal.confirm({
      title: '🚨 EMERGENCY STOP CONFIRMATION',
      content: (
        <div>
          <Alert
            message="CRITICAL ACTION"
            description="This will immediately stop the strategy and close all positions. This action cannot be undone."
            type="error"
            showIcon
            style={{ marginBottom: 16 }}
          />
          <p><strong>Strategy:</strong> {strategyInstanceId.substring(0, 8)}...</p>
          <p><strong>Reason:</strong> {reason}</p>
          <p>Type "EMERGENCY STOP" to confirm:</p>
        </div>
      ),
      icon: <ExclamationCircleOutlined style={{ color: 'red' }} />,
      okText: 'CONFIRM EMERGENCY STOP',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: async () => {
        try {
          const response = await fetch(`${apiUrl}/api/v1/nautilus/deployment/stop/${strategyInstanceId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reason })
          })

          if (response.ok) {
            message.success('Emergency stop executed')
            loadLiveStrategies()
          } else {
            message.error('Failed to execute emergency stop')
          }
        } catch (error) {
          message.error('Failed to execute emergency stop')
          console.error('Emergency stop error:', error)
        }
      }
    })
  }

  const pauseStrategy = async (strategyInstanceId: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/nautilus/deployment/pause/${strategyInstanceId}`, {
        method: 'POST'
      })

      if (response.ok) {
        message.success('Strategy paused')
        loadLiveStrategies()
      } else {
        message.error('Failed to pause strategy')
      }
    } catch (error) {
      message.error('Failed to pause strategy')
      console.error('Pause strategy error:', error)
    }
  }

  const resumeStrategy = async (strategyInstanceId: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/nautilus/deployment/resume/${strategyInstanceId}`, {
        method: 'POST'
      })

      if (response.ok) {
        message.success('Strategy resumed')
        loadLiveStrategies()
      } else {
        message.error('Failed to resume strategy')
      }
    } catch (error) {
      message.error('Failed to resume strategy')
      console.error('Resume strategy error:', error)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': case 'running': return 'success'
      case 'deploying': case 'pending_approval': return 'processing'
      case 'failed': case 'error': case 'rejected': return 'error'
      case 'paused': case 'stopped': return 'warning'
      default: return 'default'
    }
  }

  const deploymentColumns = [
    {
      title: 'Strategy',
      dataIndex: 'strategyId',
      key: 'strategyId',
      width: 120
    },
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      width: 80
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 130,
      render: (status: string) => (
        <Badge status={getStatusColor(status)} text={status.replace('_', ' ')} />
      )
    },
    {
      title: 'Risk Assessment',
      key: 'risk',
      width: 120,
      render: (_, record: DeploymentRequest) => {
        const risk = record.riskAssessment?.riskScore || 0
        return (
          <Progress
            percent={risk * 100}
            size="small"
            status={risk > 0.8 ? 'exception' : risk > 0.6 ? 'active' : 'success'}
            format={() => `${(risk * 100).toFixed(0)}%`}
          />
        )
      }
    },
    {
      title: 'Position Size',
      key: 'positionSize',
      width: 100,
      render: (_, record: DeploymentRequest) => (
        <Text>${record.proposedConfig?.positionSize?.toLocaleString()}</Text>
      )
    },
    {
      title: 'Approval',
      key: 'approval',
      width: 100,
      render: (_, record: DeploymentRequest) => (
        record.approvalRequired ? (
          <Tag color="orange">Required</Tag>
        ) : (
          <Tag color="green">Auto</Tag>
        )
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 200,
      render: (_, record: DeploymentRequest) => (
        <Space size="small">
          {record.status === 'pending_approval' && (
            <Button
              size="small"
              icon={<CheckCircleOutlined />}
              onClick={() => {
                setSelectedDeployment(record)
                setApprovalModalVisible(true)
              }}
            >
              Approve
            </Button>
          )}
          {record.status === 'approved' && (
            <Button
              size="small"
              type="primary"
              icon={<RocketOutlined />}
              onClick={() => deployStrategy(record.deploymentId!)}
            >
              Deploy
            </Button>
          )}
          {record.status === 'deployed' && (
            <Tag color="green">Live</Tag>
          )}
        </Space>
      )
    }
  ]

  const liveStrategyColumns = [
    {
      title: 'Strategy ID',
      dataIndex: 'strategyId',
      key: 'strategyId',
      width: 120
    },
    {
      title: 'State',
      dataIndex: 'state',
      key: 'state',
      width: 100,
      render: (state: string) => (
        <Badge status={getStatusColor(state)} text={state} />
      )
    },
    {
      title: 'Phase',
      dataIndex: 'currentPhase',
      key: 'currentPhase',
      width: 80,
      render: (phase: number) => `${phase + 1}/3`
    },
    {
      title: 'Position Size',
      dataIndex: 'positionSize',
      key: 'positionSize',
      width: 110,
      render: (size: number) => `$${size.toLocaleString()}`
    },
    {
      title: 'P&L',
      key: 'pnl',
      width: 120,
      render: (_, record: LiveStrategy) => {
        const totalPnL = record.realizedPnL + record.unrealizedPnL
        return (
          <Text type={totalPnL >= 0 ? 'success' : 'danger'}>
            ${totalPnL.toFixed(2)}
          </Text>
        )
      }
    },
    {
      title: 'Return',
      key: 'return',
      width: 80,
      render: (_, record: LiveStrategy) => (
        <Text type={record.performanceMetrics.totalReturn >= 0 ? 'success' : 'danger'}>
          {record.performanceMetrics.totalReturn.toFixed(2)}%
        </Text>
      )
    },
    {
      title: 'Risk',
      key: 'risk',
      width: 100,
      render: (_, record: LiveStrategy) => (
        <Tooltip title={`VaR: ${record.riskMetrics.var95.toFixed(2)}%`}>
          <Progress
            percent={Math.abs(record.riskMetrics.var95) * 10}
            size="small"
            status={Math.abs(record.riskMetrics.var95) > 5 ? 'exception' : 'normal'}
            showInfo={false}
          />
        </Tooltip>
      )
    },
    {
      title: 'Last Update',
      dataIndex: 'lastHeartbeat',
      key: 'lastHeartbeat',
      width: 120,
      render: (time: string) => dayjs(time).format('HH:mm:ss')
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 180,
      render: (_, record: LiveStrategy) => (
        <Space size="small">
          {record.state === 'running' && (
            <Button
              size="small"
              icon={<PauseOutlined />}
              onClick={() => pauseStrategy(record.strategyInstanceId)}
            >
              Pause
            </Button>
          )}
          {record.state === 'paused' && (
            <Button
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => resumeStrategy(record.strategyInstanceId)}
            >
              Resume
            </Button>
          )}
          <Button
            size="small"
            danger
            icon={<StopOutlined />}
            onClick={() => emergencyStop(record.strategyInstanceId, 'Manual stop')}
          >
            Stop
          </Button>
        </Space>
      )
    }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={[16, 16]}>
        {/* Header */}
        <Col xs={24}>
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Title level={3} style={{ margin: 0 }}>
                  <RocketOutlined style={{ marginRight: 8 }} />
                  Strategy Deployment Pipeline
                </Title>
                <Text type="secondary">
                  Deploy strategies from backtest to live trading with approval workflows
                </Text>
              </Col>
              <Col>
                <Space>
                  <Button
                    type="primary"
                    icon={<RocketOutlined />}
                    onClick={() => setCreateModalVisible(true)}
                  >
                    New Deployment
                  </Button>
                  <Button
                    danger
                    icon={<WarningOutlined />}
                    onClick={() => setEmergencyModalVisible(true)}
                  >
                    Emergency Controls
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Deployment Requests */}
        <Col xs={24}>
          <Card title="Deployment Requests">
            <Table
              dataSource={deployments}
              columns={deploymentColumns}
              rowKey="deploymentId"
              pagination={{ pageSize: 10 }}
              size="middle"
              scroll={{ x: 1000 }}
            />
          </Card>
        </Col>

        {/* Live Strategies */}
        <Col xs={24}>
          <Card title="Live Strategies" extra={<Badge count={liveStrategies.length} />}>
            <Table
              dataSource={liveStrategies}
              columns={liveStrategyColumns}
              rowKey="strategyInstanceId"
              pagination={{ pageSize: 10 }}
              size="middle"
              scroll={{ x: 1000 }}
            />
          </Card>
        </Col>
      </Row>

      {/* Create Deployment Modal */}
      <Modal
        title="Create Deployment Request"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setCreateModalVisible(false)}>
            Cancel
          </Button>,
          <Button
            key="submit"
            type="primary"
            loading={loading}
            onClick={() => form.submit()}
          >
            Create Request
          </Button>
        ]}
        width={800}
      >
        <Form form={form} layout="vertical" onFinish={createDeploymentRequest}>
          <Row gutter={16}>
            <Col xs={24} sm={12}>
              <Form.Item
                name="strategyId"
                label="Strategy ID"
                rules={[{ required: true }]}
              >
                <Input placeholder="strategy-001" />
              </Form.Item>
            </Col>
            <Col xs={24} sm={12}>
              <Form.Item
                name="version"
                label="Version"
                rules={[{ required: true }]}
              >
                <Input placeholder="v1.0.0" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col xs={24} sm={12}>
              <Form.Item
                name="backtestId"
                label="Backtest ID"
                rules={[{ required: true }]}
              >
                <Input placeholder="backtest-12345" />
              </Form.Item>
            </Col>
            <Col xs={24} sm={12}>
              <Form.Item
                name="positionSize"
                label="Position Size ($)"
                rules={[{ required: true }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={1000}
                  max={1000000}
                  formatter={value => `$ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  parser={value => value!.replace(/\$\s?|(,*)/g, '')}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col xs={24} sm={8}>
              <Form.Item
                name="maxDrawdown"
                label="Max Drawdown (%)"
                rules={[{ required: true }]}
              >
                <InputNumber style={{ width: '100%' }} min={1} max={20} />
              </Form.Item>
            </Col>
            <Col xs={24} sm={8}>
              <Form.Item
                name="stopLoss"
                label="Stop Loss (%)"
                rules={[{ required: true }]}
              >
                <InputNumber style={{ width: '100%' }} min={0.1} max={10} step={0.1} />
              </Form.Item>
            </Col>
            <Col xs={24} sm={8}>
              <Form.Item
                name="riskPerTrade"
                label="Risk Per Trade (%)"
                rules={[{ required: true }]}
              >
                <InputNumber style={{ width: '100%' }} min={0.1} max={5} step={0.1} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col xs={24} sm={12}>
              <Form.Item
                name="instruments"
                label="Instruments"
                rules={[{ required: true }]}
              >
                <Select mode="multiple" placeholder="Select instruments">
                  <Select.Option value="AAPL">AAPL</Select.Option>
                  <Select.Option value="MSFT">MSFT</Select.Option>
                  <Select.Option value="GOOGL">GOOGL</Select.Option>
                  <Select.Option value="TSLA">TSLA</Select.Option>
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} sm={12}>
              <Form.Item
                name="venues"
                label="Venues"
                rules={[{ required: true }]}
              >
                <Select mode="multiple" placeholder="Select venues">
                  <Select.Option value="NASDAQ">NASDAQ</Select.Option>
                  <Select.Option value="NYSE">NYSE</Select.Option>
                  <Select.Option value="SMART">SMART</Select.Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>

      {/* Approval Modal */}
      <Modal
        title="Approve Deployment"
        open={approvalModalVisible}
        onCancel={() => setApprovalModalVisible(false)}
        footer={[
          <Button key="reject" danger>
            Reject
          </Button>,
          <Button
            key="approve"
            type="primary"
            onClick={() => {
              if (selectedDeployment) {
                approveDeployment(selectedDeployment.deploymentId!, 'Approved via UI')
              }
            }}
          >
            Approve Deployment
          </Button>
        ]}
      >
        {selectedDeployment && (
          <div>
            <Alert
              message="Deployment Approval Required"
              description="Please review the deployment configuration and risk assessment before approval."
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />
            
            <Row gutter={16}>
              <Col span={12}>
                <Statistic title="Strategy" value={selectedDeployment.strategyId} />
              </Col>
              <Col span={12}>
                <Statistic title="Position Size" value={`$${selectedDeployment.proposedConfig?.positionSize?.toLocaleString()}`} />
              </Col>
            </Row>
            
            <div style={{ marginTop: 16 }}>
              <Text strong>Risk Assessment:</Text>
              <Progress
                percent={(selectedDeployment.riskAssessment?.riskScore || 0) * 100}
                status={(selectedDeployment.riskAssessment?.riskScore || 0) > 0.8 ? 'exception' : 'normal'}
              />
            </div>
            
            <Form.Item label="Approval Comments" style={{ marginTop: 16 }}>
              <TextArea rows={3} placeholder="Add approval comments..." />
            </Form.Item>
          </div>
        )}
      </Modal>

      {/* Emergency Controls Modal */}
      <Modal
        title="🚨 Emergency Controls"
        open={emergencyModalVisible}
        onCancel={() => setEmergencyModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setEmergencyModalVisible(false)}>
            Close
          </Button>
        ]}
        width={800}
      >
        <Alert
          message="Emergency Strategy Controls"
          description="Use these controls only in critical situations. All actions are logged and require justification."
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
        
        <div style={{ marginBottom: 16 }}>
          <Title level={5}>Quick Actions</Title>
          <Space>
            <Button 
              danger 
              icon={<StopOutlined />}
              onClick={() => {
                Modal.confirm({
                  title: 'Stop All Strategies',
                  content: 'This will stop all running strategies immediately. Continue?',
                  onOk: () => message.success('All strategies stopped')
                })
              }}
            >
              Stop All Strategies
            </Button>
            <Button 
              icon={<PauseOutlined />}
              onClick={() => {
                Modal.confirm({
                  title: 'Pause All Strategies',
                  content: 'This will pause all running strategies. Continue?',
                  onOk: () => message.success('All strategies paused')
                })
              }}
            >
              Pause All Strategies
            </Button>
          </Space>
        </div>
        
        <Table
          dataSource={liveStrategies.filter(s => s.state === 'running')}
          columns={[
            { title: 'Strategy', dataIndex: 'strategyId', key: 'strategyId' },
            { title: 'P&L', key: 'pnl', render: (_, record: LiveStrategy) => {
              const pnl = record.realizedPnL + record.unrealizedPnL
              return <Text type={pnl >= 0 ? 'success' : 'danger'}>${pnl.toFixed(2)}</Text>
            }},
            { title: 'Action', key: 'action', render: (_, record: LiveStrategy) => (
              <Button 
                size="small" 
                danger 
                onClick={() => emergencyStop(record.strategyInstanceId, 'Emergency control panel')}
              >
                Emergency Stop
              </Button>
            )}
          ]}
          size="small"
          pagination={false}
        />
      </Modal>
    </div>
  )
}

export default StrategyDeploymentPipeline