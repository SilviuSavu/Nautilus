import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Tooltip,
  Modal,
  Descriptions,
  Timeline,
  Typography,
  Row,
  Col,
  Statistic,
  Progress,
  Alert,
  Badge,
  Tabs,
  Form,
  Input,
  Select,
  Popconfirm,
  message,
  Drawer
} from 'antd';
import {
  GitlabOutlined,
  DeploymentUnitOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  PlayCircleOutlined,
  StopOutlined,
  RollbackOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  HistoryOutlined,
  UserOutlined,
  CalendarOutlined,
  TagOutlined
} from '@ant-design/icons';
import type {
  StrategyDeployment,
  DeploymentStatus,
  DeploymentApproval,
  ApproveDeploymentRequest,
  RollbackRequest,
  StrategyVersion
} from '../../types/deployment';
import { VersionComparison } from './VersionComparison';
import { RollbackManager } from './RollbackManager';

const { Text, Title } = Typography;
const { TabPane } = Tabs;

interface LifecycleManagerProps {
  strategyId?: string;
  onDeploymentSelected?: (deploymentId: string) => void;
  onVersionSelected?: (version: StrategyVersion) => void;
}

const StrategyLifecycleManager: React.FC<LifecycleManagerProps> = ({
  strategyId,
  onDeploymentSelected,
  onVersionSelected
}) => {
  const [deployments, setDeployments] = useState<StrategyDeployment[]>([]);
  const [versions, setVersions] = useState<StrategyVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDeployment, setSelectedDeployment] = useState<StrategyDeployment | null>(null);
  const [detailsVisible, setDetailsVisible] = useState(false);
  const [approvalModalVisible, setApprovalModalVisible] = useState(false);
  const [rollbackModalVisible, setRollbackModalVisible] = useState(false);
  const [comparisonVisible, setComparisonVisible] = useState(false);
  const [approvalForm] = Form.useForm();
  const [rollbackForm] = Form.useForm();

  useEffect(() => {
    if (strategyId) {
      loadLifecycleData();
    }
  }, [strategyId]);

  const loadLifecycleData = async () => {
    if (!strategyId) return;

    setLoading(true);
    try {
      // Load deployments
      const deploymentsResponse = await fetch(`/api/v1/nautilus/deployment/strategy/${strategyId}`);
      const deploymentsData = await deploymentsResponse.json();
      setDeployments(deploymentsData);

      // Load versions
      const versionsResponse = await fetch(`/api/v1/strategies/${strategyId}/versions`);
      const versionsData = await versionsResponse.json();
      setVersions(versionsData);
    } catch (error) {
      console.error('Error loading lifecycle data:', error);
      message.error('Failed to load strategy lifecycle data');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: DeploymentStatus): string => {
    switch (status) {
      case 'deployed':
      case 'running': return 'green';
      case 'pending_approval': return 'blue';
      case 'approved': return 'cyan';
      case 'deploying': return 'processing';
      case 'paused': return 'orange';
      case 'stopped': return 'default';
      case 'failed':
      case 'rolled_back': return 'red';
      default: return 'default';
    }
  };

  const getApprovalStatus = (deployment: StrategyDeployment): { approved: number; required: number; pending: string[] } => {
    const approved = deployment.approvalChain.filter(a => a.status === 'approved').length;
    const required = deployment.approvalChain.length;
    const pending = deployment.approvalChain
      .filter(a => a.status === 'pending')
      .map(a => a.requiredRole);
    
    return { approved, required, pending };
  };

  const approveDeployment = async (deploymentId: string, approved: boolean, comments?: string) => {
    try {
      const request: ApproveDeploymentRequest = {
        deploymentId,
        comments
      };

      const endpoint = approved ? 'approve' : 'reject';
      const response = await fetch(`/api/v1/nautilus/deployment/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      if (response.ok) {
        message.success(`Deployment ${approved ? 'approved' : 'rejected'} successfully`);
        loadLifecycleData();
        setApprovalModalVisible(false);
        approvalForm.resetFields();
      }
    } catch (error) {
      console.error('Approval failed:', error);
      message.error('Failed to process approval');
    }
  };

  const deployStrategy = async (deploymentId: string) => {
    try {
      const response = await fetch(`/api/v1/nautilus/deployment/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deploymentId })
      });

      if (response.ok) {
        message.success('Deployment initiated successfully');
        loadLifecycleData();
      }
    } catch (error) {
      console.error('Deployment failed:', error);
      message.error('Failed to initiate deployment');
    }
  };

  const rollbackDeployment = async (values: any) => {
    try {
      const request: RollbackRequest = {
        deploymentId: selectedDeployment!.deploymentId,
        targetVersion: values.targetVersion,
        reason: values.reason,
        immediate: values.immediate
      };

      const response = await fetch('/api/v1/nautilus/deployment/rollback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      if (response.ok) {
        message.success('Rollback initiated successfully');
        loadLifecycleData();
        setRollbackModalVisible(false);
        rollbackForm.resetFields();
      }
    } catch (error) {
      console.error('Rollback failed:', error);
      message.error('Failed to initiate rollback');
    }
  };

  const deploymentColumns = [
    {
      title: 'Deployment ID',
      dataIndex: 'deploymentId',
      render: (id: string) => (
        <Button type="link" onClick={() => onDeploymentSelected?.(id)}>
          {id.slice(0, 8)}...
        </Button>
      ),
      width: 120
    },
    {
      title: 'Version',
      dataIndex: 'version',
      render: (version: string) => <Tag>{version}</Tag>,
      width: 100
    },
    {
      title: 'Status',
      dataIndex: 'status',
      render: (status: DeploymentStatus) => (
        <Tag color={getStatusColor(status)}>
          {status.replace('_', ' ').toUpperCase()}
        </Tag>
      ),
      width: 120
    },
    {
      title: 'Approval',
      key: 'approval',
      render: (_, deployment: StrategyDeployment) => {
        const { approved, required, pending } = getApprovalStatus(deployment);
        return (
          <div>
            <Progress
              percent={Math.round((approved / required) * 100)}
              size="small"
              status={approved === required ? 'success' : 'active'}
              format={() => `${approved}/${required}`}
            />
            {pending.length > 0 && (
              <Text type="secondary" style={{ fontSize: '11px' }}>
                Pending: {pending.join(', ')}
              </Text>
            )}
          </div>
        );
      },
      width: 150
    },
    {
      title: 'Created',
      dataIndex: 'createdAt',
      render: (date: Date) => new Date(date).toLocaleDateString(),
      width: 100
    },
    {
      title: 'Created By',
      dataIndex: 'createdBy',
      render: (user: string) => (
        <div className="flex items-center space-x-1">
          <UserOutlined />
          <span>{user}</span>
        </div>
      ),
      width: 120
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, deployment: StrategyDeployment) => (
        <Space size="small">
          <Tooltip title="View Details">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedDeployment(deployment);
                setDetailsVisible(true);
              }}
            />
          </Tooltip>

          {deployment.status === 'pending_approval' && (
            <Tooltip title="Review Approval">
              <Button
                size="small"
                icon={<CheckCircleOutlined />}
                onClick={() => {
                  setSelectedDeployment(deployment);
                  setApprovalModalVisible(true);
                }}
              />
            </Tooltip>
          )}

          {deployment.status === 'approved' && (
            <Tooltip title="Deploy">
              <Button
                size="small"
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => deployStrategy(deployment.deploymentId)}
              />
            </Tooltip>
          )}

          {['deployed', 'running', 'paused'].includes(deployment.status) && (
            <Tooltip title="Rollback">
              <Button
                size="small"
                danger
                icon={<RollbackOutlined />}
                onClick={() => {
                  setSelectedDeployment(deployment);
                  setRollbackModalVisible(true);
                }}
              />
            </Tooltip>
          )}
        </Space>
      ),
      width: 150
    }
  ];

  const versionColumns = [
    {
      title: 'Version',
      dataIndex: 'version_number',
      render: (version: number, record: StrategyVersion) => (
        <Button type="link" onClick={() => onVersionSelected?.(record)}>
          v{version}
        </Button>
      )
    },
    {
      title: 'Change Summary',
      dataIndex: 'change_summary',
      ellipsis: true
    },
    {
      title: 'Created By',
      dataIndex: 'created_by',
      render: (user: string) => (
        <div className="flex items-center space-x-1">
          <UserOutlined />
          <span>{user}</span>
        </div>
      )
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      render: (date: Date) => new Date(date).toLocaleDateString()
    },
    {
      title: 'Deployments',
      dataIndex: 'deployment_results',
      render: (results: any[]) => results?.length || 0
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, version: StrategyVersion) => (
        <Space size="small">
          <Tooltip title="Compare Versions">
            <Button
              size="small"
              icon={<GitlabOutlined />}
              onClick={() => setComparisonVisible(true)}
            />
          </Tooltip>
          <Tooltip title="View History">
            <Button
              size="small"
              icon={<HistoryOutlined />}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  return (
    <div className="strategy-lifecycle-manager">
      <Card
        title={
          <div className="flex items-center space-x-2">
            <GitlabOutlined className="text-blue-600" />
            <span>Strategy Lifecycle Management</span>
          </div>
        }
        extra={
          <Button icon={<DeploymentUnitOutlined />} type="primary">
            New Deployment
          </Button>
        }
      >
        <Tabs defaultActiveKey="deployments">
          <TabPane tab="Deployments" key="deployments">
            <Table
              dataSource={deployments}
              columns={deploymentColumns}
              loading={loading}
              rowKey="deploymentId"
              pagination={{ pageSize: 10 }}
              scroll={{ x: 1000 }}
            />
          </TabPane>

          <TabPane tab="Versions" key="versions">
            <Table
              dataSource={versions}
              columns={versionColumns}
              loading={loading}
              rowKey="id"
              pagination={{ pageSize: 10 }}
            />
          </TabPane>
        </Tabs>
      </Card>

      {/* Deployment Details Modal */}
      <Modal
        title="Deployment Details"
        visible={detailsVisible}
        onCancel={() => setDetailsVisible(false)}
        footer={null}
        width={800}
      >
        {selectedDeployment && (
          <div>
            <Descriptions bordered size="small">
              <Descriptions.Item label="Deployment ID" span={3}>
                {selectedDeployment.deploymentId}
              </Descriptions.Item>
              <Descriptions.Item label="Strategy ID" span={3}>
                {selectedDeployment.strategyId}
              </Descriptions.Item>
              <Descriptions.Item label="Version" span={1}>
                {selectedDeployment.version}
              </Descriptions.Item>
              <Descriptions.Item label="Status" span={2}>
                <Tag color={getStatusColor(selectedDeployment.status)}>
                  {selectedDeployment.status.replace('_', ' ').toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Created By" span={1}>
                {selectedDeployment.createdBy}
              </Descriptions.Item>
              <Descriptions.Item label="Created At" span={2}>
                {selectedDeployment.createdAt.toLocaleString()}
              </Descriptions.Item>
              {selectedDeployment.approvedAt && (
                <>
                  <Descriptions.Item label="Approved By" span={1}>
                    {selectedDeployment.approvedBy}
                  </Descriptions.Item>
                  <Descriptions.Item label="Approved At" span={2}>
                    {selectedDeployment.approvedAt.toLocaleString()}
                  </Descriptions.Item>
                </>
              )}
            </Descriptions>

            <Card title="Approval Chain" size="small" className="mt-4">
              <Timeline>
                {selectedDeployment.approvalChain.map((approval, index) => (
                  <Timeline.Item
                    key={approval.approvalId}
                    color={approval.status === 'approved' ? 'green' : 
                           approval.status === 'rejected' ? 'red' : 'blue'}
                    dot={
                      approval.status === 'approved' ? <CheckCircleOutlined /> :
                      approval.status === 'rejected' ? <CloseCircleOutlined /> :
                      <ClockCircleOutlined />
                    }
                  >
                    <div>
                      <Text strong>{approval.approverName} ({approval.requiredRole})</Text>
                      <br />
                      <Tag color={approval.status === 'approved' ? 'green' : 
                                  approval.status === 'rejected' ? 'red' : 'blue'}>
                        {approval.status.toUpperCase()}
                      </Tag>
                      {approval.approvedAt && (
                        <Text type="secondary" className="ml-2">
                          {approval.approvedAt.toLocaleString()}
                        </Text>
                      )}
                      {approval.comments && (
                        <div className="mt-1">
                          <Text type="secondary">{approval.comments}</Text>
                        </div>
                      )}
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>
          </div>
        )}
      </Modal>

      {/* Approval Modal */}
      <Modal
        title="Review Deployment"
        visible={approvalModalVisible}
        onCancel={() => setApprovalModalVisible(false)}
        footer={null}
      >
        <Form form={approvalForm} layout="vertical">
          <Form.Item label="Comments" name="comments">
            <Input.TextArea rows={3} placeholder="Add review comments..." />
          </Form.Item>
          
          <div className="flex justify-end space-x-2">
            <Button
              danger
              onClick={() => {
                const comments = approvalForm.getFieldValue('comments');
                approveDeployment(selectedDeployment!.deploymentId, false, comments);
              }}
            >
              Reject
            </Button>
            <Button
              type="primary"
              onClick={() => {
                const comments = approvalForm.getFieldValue('comments');
                approveDeployment(selectedDeployment!.deploymentId, true, comments);
              }}
            >
              Approve
            </Button>
          </div>
        </Form>
      </Modal>

      {/* Rollback Modal */}
      <Modal
        title="Rollback Deployment"
        visible={rollbackModalVisible}
        onCancel={() => setRollbackModalVisible(false)}
        footer={null}
      >
        <Alert
          message="Warning"
          description="Rolling back will stop the current strategy and revert to the selected version. This action cannot be undone."
          type="warning"
          className="mb-4"
        />
        
        <Form form={rollbackForm} layout="vertical" onFinish={rollbackDeployment}>
          <Form.Item label="Target Version" name="targetVersion" required>
            <Select placeholder="Select version to rollback to">
              {versions.map(version => (
                <Select.Option key={version.id} value={version.version_number}>
                  v{version.version_number} - {version.change_summary}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          
          <Form.Item label="Reason" name="reason" required>
            <Input.TextArea rows={3} placeholder="Reason for rollback..." />
          </Form.Item>
          
          <Form.Item name="immediate" valuePropName="checked">
            <input type="checkbox" /> Immediate rollback (skip validation)
          </Form.Item>
          
          <div className="flex justify-end space-x-2">
            <Button onClick={() => setRollbackModalVisible(false)}>
              Cancel
            </Button>
            <Button type="primary" danger htmlType="submit">
              Rollback
            </Button>
          </div>
        </Form>
      </Modal>

      {/* Version Comparison Drawer */}
      <Drawer
        title="Version Comparison"
        placement="right"
        width={800}
        visible={comparisonVisible}
        onClose={() => setComparisonVisible(false)}
      >
        <VersionComparison versions={versions} />
      </Drawer>
    </div>
  );
};

export default StrategyLifecycleManager;