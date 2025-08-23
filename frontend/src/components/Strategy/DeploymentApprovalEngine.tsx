import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Steps,
  Button,
  Table,
  Space,
  Typography,
  Row,
  Col,
  Badge,
  Tag,
  Avatar,
  Timeline,
  Form,
  Input,
  Select,
  InputNumber,
  Switch,
  Modal,
  Alert,
  Progress,
  Tooltip,
  notification,
  Divider,
  List,
  Comment,
  Rate
} from 'antd';
import {
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  StopOutlined,
  UserOutlined,
  AuditOutlined,
  SecurityScanOutlined,
  TeamOutlined,
  SendOutlined,
  EditOutlined,
  EyeOutlined,
  CheckOutlined,
  CloseOutlined,
  WarningOutlined,
  BellOutlined,
  MailOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import type { ColumnType } from 'antd/es/table';
import type {
  DeploymentApprovalEngineProps,
  ApprovalWorkflow,
  Approval,
  ApprovalRequirement,
  ApprovalWorkflowRequest
} from './types/deploymentTypes';

const { Step } = Steps;
const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  avatar?: string;
}

const DeploymentApprovalEngine: React.FC<DeploymentApprovalEngineProps> = ({
  workflowId,
  pipelineId,
  stageId,
  requiredApprovals,
  onApprovalComplete
}) => {
  const [form] = Form.useForm();
  const [workflow, setWorkflow] = useState<ApprovalWorkflow | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showApprovalModal, setShowApprovalModal] = useState(false);
  const [selectedApproval, setSelectedApproval] = useState<Approval | null>(null);
  const [availableUsers, setAvailableUsers] = useState<User[]>([]);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [escalationTimer, setEscalationTimer] = useState<number | null>(null);
  const [notifications, setNotifications] = useState<string[]>([]);

  // Mock data for demo purposes
  const mockUsers: User[] = [
    { id: 'user1', name: 'John Doe', email: 'john.doe@example.com', role: 'senior_trader', avatar: '/avatars/john.jpg' },
    { id: 'user2', name: 'Jane Smith', email: 'jane.smith@example.com', role: 'risk_manager', avatar: '/avatars/jane.jpg' },
    { id: 'user3', name: 'Michael Brown', email: 'michael.brown@example.com', role: 'head_of_trading', avatar: '/avatars/michael.jpg' },
    { id: 'user4', name: 'Sarah Wilson', email: 'sarah.wilson@example.com', role: 'compliance', avatar: '/avatars/sarah.jpg' },
    { id: 'user5', name: 'David Chen', email: 'david.chen@example.com', role: 'cto', avatar: '/avatars/david.jpg' }
  ];

  useEffect(() => {
    setAvailableUsers(mockUsers);
    setCurrentUser(mockUsers[0]); // Mock current user
    
    if (workflowId) {
      fetchWorkflow(workflowId);
    } else {
      // Create new workflow if none provided
      createWorkflow();
    }
  }, [workflowId, pipelineId, stageId]);

  useEffect(() => {
    // Set up escalation timer if workflow has escalation settings
    if (workflow && workflow.status === 'pending') {
      const approvals = workflow.required_approvals;
      const escalationTime = approvals.find(a => a.escalation_after_hours)?.escalation_after_hours;
      
      if (escalationTime) {
        const createdTime = new Date(workflow.created_at).getTime();
        const currentTime = Date.now();
        const elapsed = (currentTime - createdTime) / (1000 * 60 * 60); // hours
        const remaining = escalationTime - elapsed;
        
        if (remaining > 0) {
          setEscalationTimer(remaining);
          
          const interval = setInterval(() => {
            setEscalationTimer(prev => {
              if (prev && prev <= 0) {
                clearInterval(interval);
                handleEscalation();
                return null;
              }
              return prev ? prev - 1/60 : null; // Decrement by 1 minute
            });
          }, 60000); // Update every minute
          
          return () => clearInterval(interval);
        }
      }
    }
  }, [workflow]);

  const fetchWorkflow = useCallback(async (workflowId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/approval/${workflowId}`);
      if (!response.ok) throw new Error('Failed to fetch workflow');
      
      const data: ApprovalWorkflow = await response.json();
      setWorkflow(data);
    } catch (error) {
      console.error('Error fetching workflow:', error);
      notification.error({
        message: 'Failed to Load Workflow',
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setLoading(false);
    }
  }, []);

  const createWorkflow = useCallback(async () => {
    if (!requiredApprovals || requiredApprovals.length === 0) return;
    
    setLoading(true);
    try {
      const request: ApprovalWorkflowRequest = {
        pipeline_id: pipelineId,
        stage_id: stageId,
        approval_requirements: requiredApprovals,
        approval_context: {
          strategy_info: 'Strategy deployment approval required',
          risk_level: 'medium'
        }
      };
      
      const response = await fetch(`${API_BASE}/api/v1/strategies/approval/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });
      
      if (!response.ok) throw new Error('Failed to create workflow');
      
      const data = await response.json();
      const newWorkflow: ApprovalWorkflow = {
        workflow_id: data.workflow_id,
        pipeline_id: pipelineId,
        stage_id: stageId,
        required_approvals: requiredApprovals,
        current_approvals: [],
        status: 'pending',
        created_at: new Date(),
        expires_at: data.expires_at ? new Date(data.expires_at) : undefined
      };
      
      setWorkflow(newWorkflow);
      
      // Send notifications
      await sendNotifications(newWorkflow);
      
      notification.success({
        message: 'Approval Workflow Created',
        description: 'Approval requests have been sent to required approvers'
      });
      
    } catch (error) {
      console.error('Error creating workflow:', error);
      notification.error({
        message: 'Failed to Create Workflow',
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setLoading(false);
    }
  }, [pipelineId, stageId, requiredApprovals]);

  const submitApproval = async (approved: boolean, comments?: string) => {
    if (!workflow || !currentUser) return;
    
    setSubmitting(true);
    try {
      const approval: Approval = {
        approval_id: `approval_${Date.now()}`,
        user_id: currentUser.id,
        role: currentUser.role,
        status: approved ? 'approved' : 'rejected',
        timestamp: new Date(),
        comments,
        signature: `${currentUser.name}_${Date.now()}` // Mock signature
      };
      
      const response = await fetch(`${API_BASE}/api/v1/strategies/approval/${workflow.workflow_id}/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(approval)
      });
      
      if (!response.ok) throw new Error('Failed to submit approval');
      
      // Update local workflow
      const updatedApprovals = [...workflow.current_approvals, approval];
      const updatedWorkflow = { ...workflow, current_approvals: updatedApprovals };
      
      // Check if workflow is complete
      const newStatus = checkWorkflowStatus(updatedWorkflow);
      updatedWorkflow.status = newStatus;
      
      setWorkflow(updatedWorkflow);
      setShowApprovalModal(false);
      
      notification.success({
        message: `Approval ${approved ? 'Granted' : 'Denied'}`,
        description: approved ? 'Your approval has been recorded' : 'Approval has been rejected'
      });
      
      if (newStatus !== 'pending') {
        onApprovalComplete?.(newStatus === 'approved');
      }
      
    } catch (error) {
      console.error('Error submitting approval:', error);
      notification.error({
        message: 'Failed to Submit Approval',
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setSubmitting(false);
    }
  };

  const checkWorkflowStatus = (workflow: ApprovalWorkflow): 'pending' | 'approved' | 'rejected' | 'expired' => {
    // Check if expired
    if (workflow.expires_at && new Date() > workflow.expires_at) {
      return 'expired';
    }
    
    // Check if any approval is rejected
    if (workflow.current_approvals.some(a => a.status === 'rejected')) {
      return 'rejected';
    }
    
    // Check if all requirements are satisfied
    for (const requirement of workflow.required_approvals) {
      const relevantApprovals = workflow.current_approvals.filter(a => 
        a.role === requirement.role && a.status === 'approved'
      );
      
      if (relevantApprovals.length < requirement.required_count) {
        return 'pending';
      }
    }
    
    return 'approved';
  };

  const sendNotifications = async (workflow: ApprovalWorkflow) => {
    const notificationPromises = workflow.required_approvals.map(async (requirement) => {
      const users = availableUsers.filter(u => u.role === requirement.role);
      
      return Promise.all(users.map(user => 
        fetch(`${API_BASE}/api/v1/notifications/send`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            recipient: user.email,
            type: 'approval_request',
            subject: 'Deployment Approval Required',
            content: `Approval required for pipeline ${workflow.pipeline_id} stage ${workflow.stage_id}`,
            workflow_id: workflow.workflow_id
          })
        })
      ));
    });
    
    try {
      await Promise.all(notificationPromises);
      setNotifications(prev => [...prev, 'Notifications sent to all required approvers']);
    } catch (error) {
      console.error('Error sending notifications:', error);
    }
  };

  const handleEscalation = async () => {
    if (!workflow) return;
    
    notification.warning({
      message: 'Approval Escalation',
      description: 'Approval deadline reached. Escalating to management.',
      duration: 10
    });
    
    // Add escalation logic here
    setNotifications(prev => [...prev, 'Approval escalated due to timeout']);
  };

  const canUserApprove = (requirement: ApprovalRequirement): boolean => {
    if (!currentUser || !workflow) return false;
    
    // Check if user role matches requirement
    if (requirement.role !== currentUser.role) return false;
    
    // Check if user has already approved
    const existingApproval = workflow.current_approvals.find(a => 
      a.user_id === currentUser.id && a.role === requirement.role
    );
    
    return !existingApproval;
  };

  const getRequirementProgress = (requirement: ApprovalRequirement): { current: number; required: number } => {
    if (!workflow) return { current: 0, required: requirement.required_count };
    
    const approvals = workflow.current_approvals.filter(a => 
      a.role === requirement.role && a.status === 'approved'
    );
    
    return { current: approvals.length, required: requirement.required_count };
  };

  const renderApprovalProgress = () => {
    if (!workflow) return null;
    
    return (
      <Card title="Approval Progress" className="mb-4">
        <Steps direction="vertical" size="small">
          {workflow.required_approvals.map((requirement, index) => {
            const progress = getRequirementProgress(requirement);
            const isComplete = progress.current >= progress.required;
            const isRejected = workflow.current_approvals.some(a => 
              a.role === requirement.role && a.status === 'rejected'
            );
            
            return (
              <Step
                key={requirement.role}
                title={
                  <div className="flex items-center justify-between">
                    <span>{requirement.role.replace('_', ' ').toUpperCase()}</span>
                    <Badge count={`${progress.current}/${progress.required}`} />
                  </div>
                }
                description={
                  <div>
                    <Progress 
                      percent={(progress.current / progress.required) * 100} 
                      size="small"
                      status={isRejected ? 'exception' : isComplete ? 'success' : 'active'}
                      showInfo={false}
                    />
                    {requirement.escalation_after_hours && (
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        Escalates after {requirement.escalation_after_hours}h
                      </Text>
                    )}
                  </div>
                }
                status={
                  isRejected ? 'error' :
                  isComplete ? 'finish' : 
                  'process'
                }
                icon={
                  isRejected ? <CloseOutlined /> :
                  isComplete ? <CheckOutlined /> :
                  <ClockCircleOutlined />
                }
              />
            );
          })}
        </Steps>
        
        {escalationTimer && escalationTimer > 0 && (
          <Alert
            message="Escalation Warning"
            description={`Approval will escalate in ${Math.floor(escalationTimer)}h ${Math.floor((escalationTimer % 1) * 60)}m`}
            type="warning"
            showIcon
            className="mt-4"
          />
        )}
      </Card>
    );
  };

  const renderApprovalHistory = () => {
    if (!workflow || workflow.current_approvals.length === 0) return null;
    
    return (
      <Card title="Approval History" className="mb-4">
        <Timeline>
          {workflow.current_approvals.map((approval, index) => {
            const user = availableUsers.find(u => u.id === approval.user_id);
            
            return (
              <Timeline.Item
                key={approval.approval_id}
                color={approval.status === 'approved' ? 'green' : 'red'}
                dot={approval.status === 'approved' ? <CheckCircleOutlined /> : <CloseOutlined />}
              >
                <div>
                  <Space>
                    <Avatar size="small" src={user?.avatar} icon={<UserOutlined />} />
                    <Text strong>{user?.name || 'Unknown User'}</Text>
                    <Tag color={approval.status === 'approved' ? 'success' : 'error'}>
                      {approval.status.toUpperCase()}
                    </Tag>
                  </Space>
                  <div className="mt-2">
                    <Text type="secondary">
                      {dayjs(approval.timestamp).format('YYYY-MM-DD HH:mm:ss')}
                    </Text>
                  </div>
                  {approval.comments && (
                    <div className="mt-2">
                      <Text>{approval.comments}</Text>
                    </div>
                  )}
                </div>
              </Timeline.Item>
            );
          })}
        </Timeline>
      </Card>
    );
  };

  const renderActionButtons = () => {
    if (!workflow || !currentUser) return null;
    
    const canApprove = workflow.required_approvals.some(req => canUserApprove(req));
    const isComplete = workflow.status !== 'pending';
    
    if (isComplete) {
      return (
        <Alert
          message={`Workflow ${workflow.status.toUpperCase()}`}
          type={workflow.status === 'approved' ? 'success' : 'error'}
          showIcon
        />
      );
    }
    
    if (!canApprove) {
      return (
        <Alert
          message="Waiting for Approval"
          description="You are not authorized to approve this workflow or have already submitted your approval."
          type="info"
          showIcon
        />
      );
    }
    
    return (
      <Space>
        <Button
          type="primary"
          icon={<CheckOutlined />}
          onClick={() => setShowApprovalModal(true)}
        >
          Review & Approve
        </Button>
        <Button
          danger
          icon={<CloseOutlined />}
          onClick={() => {
            Modal.confirm({
              title: 'Reject Approval',
              content: 'Are you sure you want to reject this deployment?',
              onOk: () => submitApproval(false, 'Rejected by user')
            });
          }}
        >
          Reject
        </Button>
      </Space>
    );
  };

  if (loading) {
    return <Card loading={true} />;
  }

  if (!workflow) {
    return (
      <Card>
        <Alert
          message="No Approval Workflow"
          description="No approval workflow found for this deployment."
          type="info"
          showIcon
        />
      </Card>
    );
  }

  return (
    <div className="deployment-approval-engine">
      <Card
        title={
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <AuditOutlined />
              <span>Deployment Approval</span>
              <Tag color={
                workflow.status === 'approved' ? 'success' :
                workflow.status === 'rejected' ? 'error' :
                workflow.status === 'expired' ? 'warning' : 'processing'
              }>
                {workflow.status.toUpperCase()}
              </Tag>
            </div>
            <Space>
              <Text type="secondary">ID: {workflow.workflow_id}</Text>
              {workflow.expires_at && (
                <Tooltip title="Expiration Date">
                  <Text type="secondary">
                    Expires: {dayjs(workflow.expires_at).format('MM/DD HH:mm')}
                  </Text>
                </Tooltip>
              )}
            </Space>
          </div>
        }
      >
        <Row gutter={16}>
          <Col span={16}>
            {renderApprovalProgress()}
            {renderApprovalHistory()}
            
            {notifications.length > 0 && (
              <Card title="Notifications" size="small" className="mb-4">
                <List
                  size="small"
                  dataSource={notifications}
                  renderItem={(item, index) => (
                    <List.Item key={index}>
                      <BellOutlined className="mr-2" />
                      {item}
                    </List.Item>
                  )}
                />
              </Card>
            )}
          </Col>
          
          <Col span={8}>
            <Card title="Workflow Details" size="small" className="mb-4">
              <div className="space-y-2">
                <div>
                  <Text strong>Pipeline: </Text>
                  <Text code>{workflow.pipeline_id}</Text>
                </div>
                <div>
                  <Text strong>Stage: </Text>
                  <Text>{workflow.stage_id}</Text>
                </div>
                <div>
                  <Text strong>Created: </Text>
                  <Text>{dayjs(workflow.created_at).format('MM/DD HH:mm')}</Text>
                </div>
                <div>
                  <Text strong>Status: </Text>
                  <Tag color={workflow.status === 'approved' ? 'success' : 'processing'}>
                    {workflow.status.toUpperCase()}
                  </Tag>
                </div>
              </div>
            </Card>
            
            <Card title="Required Approvers" size="small" className="mb-4">
              {workflow.required_approvals.map((requirement, index) => {
                const users = availableUsers.filter(u => u.role === requirement.role);
                const progress = getRequirementProgress(requirement);
                
                return (
                  <div key={index} className="mb-3">
                    <div className="flex justify-between items-center mb-1">
                      <Text strong>{requirement.role.replace('_', ' ').toUpperCase()}</Text>
                      <Badge count={`${progress.current}/${progress.required}`} />
                    </div>
                    <div>
                      {users.map(user => (
                        <div key={user.id} className="flex items-center space-x-2 mb-1">
                          <Avatar size="small" src={user.avatar} icon={<UserOutlined />} />
                          <Text style={{ fontSize: '12px' }}>{user.name}</Text>
                          {workflow.current_approvals.find(a => 
                            a.user_id === user.id && a.role === requirement.role
                          ) && (
                            <CheckCircleOutlined style={{ color: '#52c41a' }} />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </Card>
            
            <div>{renderActionButtons()}</div>
          </Col>
        </Row>
      </Card>

      {/* Approval Modal */}
      <Modal
        title="Submit Approval"
        open={showApprovalModal}
        onCancel={() => setShowApprovalModal(false)}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={(values) => submitApproval(values.decision === 'approve', values.comments)}
        >
          <Form.Item
            label="Decision"
            name="decision"
            rules={[{ required: true, message: 'Please make a decision' }]}
          >
            <Select placeholder="Select your decision">
              <Option value="approve">Approve</Option>
              <Option value="reject">Reject</Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            label="Comments"
            name="comments"
            rules={[{ required: true, message: 'Please provide comments' }]}
          >
            <TextArea 
              rows={4} 
              placeholder="Please provide your reasoning for this decision..."
            />
          </Form.Item>
          
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={submitting}>
                Submit Approval
              </Button>
              <Button onClick={() => setShowApprovalModal(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default DeploymentApprovalEngine;