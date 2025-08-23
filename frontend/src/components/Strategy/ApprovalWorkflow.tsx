/**
 * Approval Workflow Component
 * Multi-stage approval process with role-based permissions
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Tag,
  Space,
  Modal,
  Form,
  Input,
  Select,
  Typography,
  Steps,
  Timeline,
  Avatar,
  Rate,
  Alert,
  Tooltip,
  Badge,
  Divider,
  List,
  Progress,
  Drawer,
  Tabs
} from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  UserOutlined,
  TeamOutlined,
  SafetyOutlined,
  ExclamationCircleOutlined,
  EyeOutlined,
  CommentOutlined,
  LikeOutlined,
  DislikeOutlined,
  CrownOutlined,
  AuditOutlined,
  BellOutlined
} from '@ant-design/icons';
import { useStrategyDeployment } from '../../hooks/strategy/useStrategyDeployment';

const { Title, Text, Paragraph } = Typography;
const { Step } = Steps;
const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;

interface ApprovalWorkflowProps {
  deploymentRequestId?: string;
  strategyId?: string;
  onApprovalComplete?: (approved: boolean) => void;
}

interface ApprovalStage {
  stageId: string;
  name: string;
  description: string;
  requiredRole: string;
  approvers: string[];
  requiredApprovals: number;
  currentApprovals: number;
  status: 'pending' | 'approved' | 'rejected' | 'skipped';
  order: number;
  autoApprove?: boolean;
  conditions?: {
    field: string;
    operator: string;
    value: any;
  }[];
}

interface ApprovalComment {
  commentId: string;
  userId: string;
  userName: string;
  userRole: string;
  comment: string;
  decision: 'approve' | 'reject' | 'comment';
  timestamp: Date;
  stageId: string;
}

interface ApprovalWorkflowData {
  workflowId: string;
  requestId: string;
  strategyId: string;
  strategyName: string;
  version: string;
  environment: string;
  status: 'pending' | 'approved' | 'rejected' | 'cancelled';
  currentStage: number;
  stages: ApprovalStage[];
  comments: ApprovalComment[];
  requestedBy: string;
  requestedAt: Date;
  completedAt?: Date;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  metadata: Record<string, any>;
}

export const ApprovalWorkflow: React.FC<ApprovalWorkflowProps> = ({
  deploymentRequestId,
  strategyId,
  onApprovalComplete
}) => {
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(deploymentRequestId || null);
  const [approvalModalVisible, setApprovalModalVisible] = useState(false);
  const [commentModalVisible, setCommentModalVisible] = useState(false);
  const [workflowDetailsVisible, setWorkflowDetailsVisible] = useState(false);
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  const [selectedAction, setSelectedAction] = useState<'approve' | 'reject' | 'comment'>('approve');

  const [approvalForm] = Form.useForm();
  const [commentForm] = Form.useForm();

  // Mock data - in real implementation, this would come from the API
  const [workflows, setWorkflows] = useState<ApprovalWorkflowData[]>([
    {
      workflowId: 'wf-001',
      requestId: 'req-001',
      strategyId: 'strategy-1',
      strategyName: 'EMA Cross Strategy',
      version: 'v1.2.0',
      environment: 'production',
      status: 'pending',
      currentStage: 1,
      requestedBy: 'john.doe',
      requestedAt: new Date('2024-01-15T10:30:00Z'),
      urgency: 'high',
      riskLevel: 'medium',
      metadata: {
        deploymentType: 'blue_green',
        estimatedDowntime: '5 minutes',
        affectedUsers: 1000
      },
      stages: [
        {
          stageId: 'stage-1',
          name: 'Technical Review',
          description: 'Code review and technical validation',
          requiredRole: 'developer',
          approvers: ['jane.smith', 'bob.wilson'],
          requiredApprovals: 2,
          currentApprovals: 2,
          status: 'approved',
          order: 1
        },
        {
          stageId: 'stage-2',
          name: 'Security Review',
          description: 'Security assessment and risk evaluation',
          requiredRole: 'security',
          approvers: ['alice.cooper', 'mike.jones'],
          requiredApprovals: 1,
          currentApprovals: 0,
          status: 'pending',
          order: 2
        },
        {
          stageId: 'stage-3',
          name: 'Business Approval',
          description: 'Business stakeholder approval',
          requiredRole: 'manager',
          approvers: ['sarah.johnson'],
          requiredApprovals: 1,
          currentApprovals: 0,
          status: 'pending',
          order: 3
        },
        {
          stageId: 'stage-4',
          name: 'Final Sign-off',
          description: 'Final deployment authorization',
          requiredRole: 'admin',
          approvers: ['admin.user'],
          requiredApprovals: 1,
          currentApprovals: 0,
          status: 'pending',
          order: 4
        }
      ],
      comments: [
        {
          commentId: 'comment-1',
          userId: 'jane.smith',
          userName: 'Jane Smith',
          userRole: 'Senior Developer',
          comment: 'Code looks good, all tests pass. Performance improvements are solid.',
          decision: 'approve',
          timestamp: new Date('2024-01-15T11:00:00Z'),
          stageId: 'stage-1'
        },
        {
          commentId: 'comment-2',
          userId: 'bob.wilson',
          userName: 'Bob Wilson',
          userRole: 'Lead Developer',
          comment: 'Architecture is sound, deployment strategy is appropriate.',
          decision: 'approve',
          timestamp: new Date('2024-01-15T11:15:00Z'),
          stageId: 'stage-1'
        }
      ]
    }
  ]);

  const {
    deploymentRequests,
    approveDeploymentRequest,
    loading
  } = useStrategyDeployment();

  // Handle approval action
  const handleApproval = async (values: any) => {
    try {
      const workflow = workflows.find(w => w.workflowId === selectedWorkflow);
      if (!workflow) return;

      const success = await approveDeploymentRequest(
        workflow.requestId,
        'current-user',
        selectedAction === 'approve'
      );

      if (success) {
        // Update local state
        setWorkflows(prev => prev.map(w => {
          if (w.workflowId === selectedWorkflow) {
            const updatedStages = w.stages.map(stage => {
              if (stage.stageId === selectedStage) {
                return {
                  ...stage,
                  currentApprovals: selectedAction === 'approve' 
                    ? stage.currentApprovals + 1 
                    : stage.currentApprovals,
                  status: selectedAction === 'approve' && stage.currentApprovals + 1 >= stage.requiredApprovals
                    ? 'approved' as const
                    : selectedAction === 'reject' ? 'rejected' as const : stage.status
                };
              }
              return stage;
            });

            // Add comment
            const newComment: ApprovalComment = {
              commentId: `comment-${Date.now()}`,
              userId: 'current-user',
              userName: 'Current User',
              userRole: 'User',
              comment: values.comment || '',
              decision: selectedAction,
              timestamp: new Date(),
              stageId: selectedStage!
            };

            return {
              ...w,
              stages: updatedStages,
              comments: [...w.comments, newComment]
            };
          }
          return w;
        }));

        onApprovalComplete?.(selectedAction === 'approve');
      }

      setApprovalModalVisible(false);
      setCommentModalVisible(false);
      approvalForm.resetFields();
      commentForm.resetFields();
    } catch (error) {
      console.error('Failed to process approval:', error);
    }
  };

  // Get stage status icon
  const getStageStatusIcon = (status: string) => {
    switch (status) {
      case 'approved':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'rejected':
        return <CloseCircleOutlined style={{ color: '#f5222d' }} />;
      case 'pending':
        return <ClockCircleOutlined style={{ color: '#faad14' }} />;
      case 'skipped':
        return <ExclamationCircleOutlined style={{ color: '#d9d9d9' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  // Get role icon
  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'admin':
        return <CrownOutlined style={{ color: '#722ed1' }} />;
      case 'manager':
        return <TeamOutlined style={{ color: '#13c2c2' }} />;
      case 'security':
        return <SafetyOutlined style={{ color: '#f5222d' }} />;
      case 'developer':
        return <UserOutlined style={{ color: '#1890ff' }} />;
      default:
        return <UserOutlined />;
    }
  };

  // Get urgency color
  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'critical':
        return '#f5222d';
      case 'high':
        return '#fa8c16';
      case 'medium':
        return '#fadb14';
      case 'low':
        return '#52c41a';
      default:
        return '#d9d9d9';
    }
  };

  // Get risk level color
  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical':
        return '#f5222d';
      case 'high':
        return '#fa8c16';
      case 'medium':
        return '#fadb14';
      case 'low':
        return '#52c41a';
      default:
        return '#d9d9d9';
    }
  };

  const selectedWorkflowData = workflows.find(w => w.workflowId === selectedWorkflow);

  // Workflow columns
  const workflowColumns = [
    {
      title: 'Request',
      key: 'request',
      render: (_: any, record: ApprovalWorkflowData) => (
        <div>
          <Button
            type="link"
            onClick={() => {
              setSelectedWorkflow(record.workflowId);
              setWorkflowDetailsVisible(true);
            }}
            style={{ padding: 0 }}
          >
            <Text strong>{record.strategyName}</Text>
          </Button>
          <br />
          <Text type="secondary">{record.version} → {record.environment}</Text>
        </div>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const color = status === 'approved' ? 'green' : 
                     status === 'rejected' ? 'red' : 'orange';
        return <Tag color={color}>{status}</Tag>;
      }
    },
    {
      title: 'Progress',
      key: 'progress',
      render: (_: any, record: ApprovalWorkflowData) => {
        const approvedStages = record.stages.filter(s => s.status === 'approved').length;
        const totalStages = record.stages.length;
        return (
          <Progress
            percent={Math.round((approvedStages / totalStages) * 100)}
            size="small"
            format={() => `${approvedStages}/${totalStages}`}
          />
        );
      }
    },
    {
      title: 'Urgency',
      dataIndex: 'urgency',
      key: 'urgency',
      render: (urgency: string) => (
        <Tag color={getUrgencyColor(urgency)}>{urgency}</Tag>
      )
    },
    {
      title: 'Risk',
      dataIndex: 'riskLevel',
      key: 'riskLevel',
      render: (riskLevel: string) => (
        <Tag color={getRiskLevelColor(riskLevel)}>{riskLevel}</Tag>
      )
    },
    {
      title: 'Requested By',
      dataIndex: 'requestedBy',
      key: 'requestedBy',
      render: (user: string) => (
        <Space>
          <Avatar size="small" icon={<UserOutlined />} />
          <Text>{user}</Text>
        </Space>
      )
    },
    {
      title: 'Requested At',
      dataIndex: 'requestedAt',
      key: 'requestedAt',
      render: (date: Date) => new Date(date).toLocaleString()
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: ApprovalWorkflowData) => {
        const currentStage = record.stages[record.currentStage];
        const canApprove = currentStage && 
          currentStage.status === 'pending' &&
          currentStage.approvers.includes('current-user'); // Mock user check

        return (
          <Space>
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedWorkflow(record.workflowId);
                setWorkflowDetailsVisible(true);
              }}
            >
              View
            </Button>
            {canApprove && (
              <>
                <Button
                  size="small"
                  type="primary"
                  icon={<CheckCircleOutlined />}
                  onClick={() => {
                    setSelectedWorkflow(record.workflowId);
                    setSelectedStage(currentStage.stageId);
                    setSelectedAction('approve');
                    setApprovalModalVisible(true);
                  }}
                >
                  Approve
                </Button>
                <Button
                  size="small"
                  danger
                  icon={<CloseCircleOutlined />}
                  onClick={() => {
                    setSelectedWorkflow(record.workflowId);
                    setSelectedStage(currentStage.stageId);
                    setSelectedAction('reject');
                    setApprovalModalVisible(true);
                  }}
                >
                  Reject
                </Button>
              </>
            )}
            <Button
              size="small"
              icon={<CommentOutlined />}
              onClick={() => {
                setSelectedWorkflow(record.workflowId);
                setSelectedStage(currentStage?.stageId || record.stages[0].stageId);
                setSelectedAction('comment');
                setCommentModalVisible(true);
              }}
            >
              Comment
            </Button>
          </Space>
        );
      }
    }
  ];

  return (
    <div className="approval-workflow">
      <div style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <Title level={3}>
              <AuditOutlined /> Approval Workflow
            </Title>
          </Col>
          <Col>
            <Space>
              <Badge count={workflows.filter(w => w.status === 'pending').length}>
                <Button icon={<BellOutlined />}>
                  Pending Approvals
                </Button>
              </Badge>
            </Space>
          </Col>
        </Row>
      </div>

      {/* Summary Cards */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <Text type="secondary">Pending</Text>
              <Title level={2} style={{ margin: 0, color: '#faad14' }}>
                {workflows.filter(w => w.status === 'pending').length}
              </Title>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <Text type="secondary">Approved</Text>
              <Title level={2} style={{ margin: 0, color: '#52c41a' }}>
                {workflows.filter(w => w.status === 'approved').length}
              </Title>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <Text type="secondary">Rejected</Text>
              <Title level={2} style={{ margin: 0, color: '#f5222d' }}>
                {workflows.filter(w => w.status === 'rejected').length}
              </Title>
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <div style={{ textAlign: 'center' }}>
              <Text type="secondary">High Priority</Text>
              <Title level={2} style={{ margin: 0, color: '#fa8c16' }}>
                {workflows.filter(w => w.urgency === 'high' || w.urgency === 'critical').length}
              </Title>
            </div>
          </Card>
        </Col>
      </Row>

      {/* Workflows Table */}
      <Card title="Approval Requests">
        <Table
          dataSource={workflows}
          columns={workflowColumns}
          loading={loading}
          rowKey="workflowId"
          pagination={{ pageSize: 20 }}
          rowClassName={(record) => 
            record.urgency === 'critical' ? 'critical-row' :
            record.urgency === 'high' ? 'high-priority-row' : ''
          }
        />
      </Card>

      {/* Approval Modal */}
      <Modal
        title={`${selectedAction === 'approve' ? 'Approve' : 'Reject'} Request`}
        open={approvalModalVisible}
        onCancel={() => setApprovalModalVisible(false)}
        footer={null}
        width={600}
      >
        {selectedWorkflowData && (
          <div>
            <Alert
              message={`${selectedAction === 'approve' ? 'Approving' : 'Rejecting'} deployment request`}
              description={`${selectedWorkflowData.strategyName} ${selectedWorkflowData.version} to ${selectedWorkflowData.environment}`}
              type={selectedAction === 'approve' ? 'success' : 'error'}
              showIcon
              style={{ marginBottom: 16 }}
            />

            <Form
              form={approvalForm}
              layout="vertical"
              onFinish={handleApproval}
            >
              <Form.Item
                name="comment"
                label="Comment"
                rules={[{ required: selectedAction === 'reject', message: 'Comment is required for rejection' }]}
              >
                <TextArea
                  rows={4}
                  placeholder={`Please provide ${selectedAction === 'approve' ? 'approval' : 'rejection'} comments...`}
                />
              </Form.Item>

              {selectedAction === 'approve' && (
                <Form.Item
                  name="conditions"
                  label="Approval Conditions"
                >
                  <Select mode="multiple" placeholder="Select any conditions">
                    <Option value="monitor_closely">Monitor closely</Option>
                    <Option value="rollback_ready">Rollback ready</Option>
                    <Option value="notify_on_issues">Notify on issues</Option>
                    <Option value="limited_rollout">Limited rollout</Option>
                  </Select>
                </Form.Item>
              )}

              <Form.Item>
                <Space>
                  <Button
                    type={selectedAction === 'approve' ? 'primary' : 'default'}
                    danger={selectedAction === 'reject'}
                    htmlType="submit"
                    loading={loading}
                    icon={selectedAction === 'approve' ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                  >
                    {selectedAction === 'approve' ? 'Approve' : 'Reject'}
                  </Button>
                  <Button onClick={() => setApprovalModalVisible(false)}>
                    Cancel
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          </div>
        )}
      </Modal>

      {/* Comment Modal */}
      <Modal
        title="Add Comment"
        open={commentModalVisible}
        onCancel={() => setCommentModalVisible(false)}
        footer={null}
      >
        <Form
          form={commentForm}
          layout="vertical"
          onFinish={handleApproval}
        >
          <Form.Item
            name="comment"
            label="Comment"
            rules={[{ required: true, message: 'Please enter your comment' }]}
          >
            <TextArea
              rows={4}
              placeholder="Enter your comment or feedback..."
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                Add Comment
              </Button>
              <Button onClick={() => setCommentModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Workflow Details Drawer */}
      <Drawer
        title="Approval Workflow Details"
        placement="right"
        width={800}
        open={workflowDetailsVisible}
        onClose={() => setWorkflowDetailsVisible(false)}
      >
        {selectedWorkflowData && (
          <div>
            <Tabs defaultActiveKey="overview">
              <TabPane tab="Overview" key="overview">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Card size="small" title="Request Information">
                    <Row gutter={16}>
                      <Col span={8}>
                        <Text strong>Strategy:</Text><br />
                        <Text>{selectedWorkflowData.strategyName}</Text>
                      </Col>
                      <Col span={8}>
                        <Text strong>Version:</Text><br />
                        <Tag color="blue">{selectedWorkflowData.version}</Tag>
                      </Col>
                      <Col span={8}>
                        <Text strong>Environment:</Text><br />
                        <Tag color="green">{selectedWorkflowData.environment}</Tag>
                      </Col>
                    </Row>
                    <Divider />
                    <Row gutter={16}>
                      <Col span={6}>
                        <Text strong>Status:</Text><br />
                        <Tag color={selectedWorkflowData.status === 'approved' ? 'green' : 
                                   selectedWorkflowData.status === 'rejected' ? 'red' : 'orange'}>
                          {selectedWorkflowData.status}
                        </Tag>
                      </Col>
                      <Col span={6}>
                        <Text strong>Urgency:</Text><br />
                        <Tag color={getUrgencyColor(selectedWorkflowData.urgency)}>
                          {selectedWorkflowData.urgency}
                        </Tag>
                      </Col>
                      <Col span={6}>
                        <Text strong>Risk Level:</Text><br />
                        <Tag color={getRiskLevelColor(selectedWorkflowData.riskLevel)}>
                          {selectedWorkflowData.riskLevel}
                        </Tag>
                      </Col>
                      <Col span={6}>
                        <Text strong>Requested By:</Text><br />
                        <Space>
                          <Avatar size="small" icon={<UserOutlined />} />
                          <Text>{selectedWorkflowData.requestedBy}</Text>
                        </Space>
                      </Col>
                    </Row>
                  </Card>

                  <Card size="small" title="Approval Progress">
                    <Steps
                      current={selectedWorkflowData.currentStage}
                      status={selectedWorkflowData.status === 'rejected' ? 'error' : 'process'}
                    >
                      {selectedWorkflowData.stages.map((stage, index) => (
                        <Step
                          key={stage.stageId}
                          title={stage.name}
                          description={
                            <div>
                              <Space>
                                {getRoleIcon(stage.requiredRole)}
                                <Text type="secondary">{stage.requiredRole}</Text>
                              </Space>
                              <br />
                              <Progress
                                percent={Math.round((stage.currentApprovals / stage.requiredApprovals) * 100)}
                                size="small"
                                format={() => `${stage.currentApprovals}/${stage.requiredApprovals}`}
                              />
                            </div>
                          }
                          icon={getStageStatusIcon(stage.status)}
                        />
                      ))}
                    </Steps>
                  </Card>

                  <Card size="small" title="Metadata">
                    <Row gutter={16}>
                      {Object.entries(selectedWorkflowData.metadata).map(([key, value]) => (
                        <Col span={8} key={key}>
                          <Text strong>{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}:</Text><br />
                          <Text>{String(value)}</Text>
                        </Col>
                      ))}
                    </Row>
                  </Card>
                </Space>
              </TabPane>

              <TabPane tab="Stages" key="stages">
                <List
                  dataSource={selectedWorkflowData.stages}
                  renderItem={(stage) => (
                    <List.Item
                      actions={[
                        stage.status === 'pending' && stage.approvers.includes('current-user') && (
                          <Space key="actions">
                            <Button
                              size="small"
                              type="primary"
                              icon={<CheckCircleOutlined />}
                              onClick={() => {
                                setSelectedStage(stage.stageId);
                                setSelectedAction('approve');
                                setApprovalModalVisible(true);
                              }}
                            >
                              Approve
                            </Button>
                            <Button
                              size="small"
                              danger
                              icon={<CloseCircleOutlined />}
                              onClick={() => {
                                setSelectedStage(stage.stageId);
                                setSelectedAction('reject');
                                setApprovalModalVisible(true);
                              }}
                            >
                              Reject
                            </Button>
                          </Space>
                        )
                      ].filter(Boolean)}
                    >
                      <List.Item.Meta
                        avatar={
                          <Avatar
                            style={{
                              backgroundColor: stage.status === 'approved' ? '#52c41a' :
                                            stage.status === 'rejected' ? '#f5222d' :
                                            stage.status === 'pending' ? '#faad14' : '#d9d9d9'
                            }}
                            icon={getStageStatusIcon(stage.status)}
                          />
                        }
                        title={
                          <Space>
                            <Text strong>{stage.name}</Text>
                            <Tag color={stage.status === 'approved' ? 'green' : 
                                       stage.status === 'rejected' ? 'red' : 'orange'}>
                              {stage.status}
                            </Tag>
                            {getRoleIcon(stage.requiredRole)}
                            <Text type="secondary">{stage.requiredRole}</Text>
                          </Space>
                        }
                        description={
                          <div>
                            <Paragraph>{stage.description}</Paragraph>
                            <Space>
                              <Text type="secondary">
                                Approvals: {stage.currentApprovals}/{stage.requiredApprovals}
                              </Text>
                              <Divider type="vertical" />
                              <Text type="secondary">
                                Approvers: {stage.approvers.join(', ')}
                              </Text>
                            </Space>
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />
              </TabPane>

              <TabPane tab="Comments" key="comments">
                <List
                  dataSource={selectedWorkflowData.comments}
                  renderItem={(comment) => (
                    <Comment
                      author={
                        <Space>
                          <Text strong>{comment.userName}</Text>
                          <Text type="secondary">({comment.userRole})</Text>
                          <Tag color={comment.decision === 'approve' ? 'green' : 
                                     comment.decision === 'reject' ? 'red' : 'blue'}>
                            {comment.decision}
                          </Tag>
                        </Space>
                      }
                      avatar={<Avatar icon={<UserOutlined />} />}
                      content={<Paragraph>{comment.comment}</Paragraph>}
                      datetime={
                        <Tooltip title={new Date(comment.timestamp).toLocaleString()}>
                          <span>{new Date(comment.timestamp).toLocaleDateString()}</span>
                        </Tooltip>
                      }
                      actions={[
                        <span key="like">
                          <LikeOutlined /> Like
                        </span>,
                        <span key="reply">Reply</span>
                      ]}
                    />
                  )}
                />
              </TabPane>
            </Tabs>
          </div>
        )}
      </Drawer>

      <style jsx global>{`
        .critical-row {
          background-color: #fff2f0;
        }
        .high-priority-row {
          background-color: #fff7e6;
        }
      `}</style>
    </div>
  );
};