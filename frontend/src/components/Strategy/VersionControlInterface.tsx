/**
 * Version Control Interface
 * Git-like version control for strategies with branching and merging
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Tag,
  Space,
  Modal,
  Form,
  Input,
  Select,
  Typography,
  Tree,
  Timeline,
  Tooltip,
  Drawer,
  Alert,
  Tabs,
  List,
  Divider,
  Progress,
  Avatar,
  Comment,
  Rate,
  message
} from 'antd';
import {
  GitlabOutlined,
  TagOutlined,
  MergeOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  DiffOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  UserOutlined,
  HistoryOutlined,
  CodeOutlined
} from '@ant-design/icons';
import { useVersionControl } from '../../hooks/strategy/useVersionControl';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;

interface VersionControlInterfaceProps {
  strategyId: string;
  currentBranch?: string;
  onVersionSelect?: (versionId: string) => void;
  onBranchChange?: (branchName: string) => void;
}

export const VersionControlInterface: React.FC<VersionControlInterfaceProps> = ({
  strategyId,
  currentBranch = 'main',
  onVersionSelect,
  onBranchChange
}) => {
  const [activeTab, setActiveTab] = useState('versions');
  const [selectedVersion, setSelectedVersion] = useState<string | null>(null);
  const [selectedMergeRequest, setSelectedMergeRequest] = useState<string | null>(null);
  const [newBranchModalVisible, setNewBranchModalVisible] = useState(false);
  const [newVersionModalVisible, setNewVersionModalVisible] = useState(false);
  const [mergeRequestModalVisible, setMergeRequestModalVisible] = useState(false);
  const [tagModalVisible, setTagModalVisible] = useState(false);
  const [compareModalVisible, setCompareModalVisible] = useState(false);
  const [versionDetailsVisible, setVersionDetailsVisible] = useState(false);
  const [compareVersionsState, setCompareVersionsState] = useState<{base: string; compare: string}>({base: '', compare: ''});

  const [branchForm] = Form.useForm();
  const [versionForm] = Form.useForm();
  const [mergeForm] = Form.useForm();
  const [tagForm] = Form.useForm();

  const {
    versions,
    branches,
    mergeRequests,
    tags,
    currentBranch: activeBranch,
    loading,
    error,
    createVersion,
    createBranch,
    switchBranch,
    createMergeRequest,
    approveMergeRequest,
    mergeBranches,
    compareVersions,
    createTag,
    getVersion,
    getBranch,
    getMergeRequest,
    getLatestVersion,
    fetchVersions,
    fetchBranches,
    fetchMergeRequests,
    fetchTags,
    setCurrentBranch
  } = useVersionControl();

  // Initialize data
  useEffect(() => {
    fetchVersions(strategyId, currentBranch);
    fetchBranches(strategyId);
    fetchMergeRequests(strategyId);
    fetchTags(strategyId);
  }, [strategyId, currentBranch, fetchVersions, fetchBranches, fetchMergeRequests, fetchTags]);

  // Handle branch creation
  const handleCreateBranch = async (values: any) => {
    try {
      await createBranch(
        strategyId,
        values.name,
        values.baseVersion,
        values.description,
        'user'
      );
      setNewBranchModalVisible(false);
      branchForm.resetFields();
      message.success('Branch created successfully');
    } catch (error) {
      message.error(`Failed to create branch: ${error}`);
    }
  };

  // Handle version creation
  const handleCreateVersion = async (values: any) => {
    try {
      await createVersion(
        strategyId,
        values.strategyCode,
        values.strategyConfig || {},
        values.message,
        values.branch || currentBranch,
        'user'
      );
      setNewVersionModalVisible(false);
      versionForm.resetFields();
      message.success('Version created successfully');
    } catch (error) {
      message.error(`Failed to create version: ${error}`);
    }
  };

  // Handle merge request creation
  const handleCreateMergeRequest = async (values: any) => {
    try {
      await createMergeRequest(
        strategyId,
        values.sourceBranch,
        values.targetBranch,
        values.title,
        values.description,
        values.reviewers || [],
        'user'
      );
      setMergeRequestModalVisible(false);
      mergeForm.resetFields();
      message.success('Merge request created successfully');
    } catch (error) {
      message.error(`Failed to create merge request: ${error}`);
    }
  };

  // Handle tag creation
  const handleCreateTag = async (values: any) => {
    try {
      await createTag(
        values.versionId,
        values.name,
        values.description,
        values.isRelease || false,
        'user'
      );
      setTagModalVisible(false);
      tagForm.resetFields();
      message.success('Tag created successfully');
    } catch (error) {
      message.error(`Failed to create tag: ${error}`);
    }
  };

  // Handle version comparison
  const handleCompareVersions = async () => {
    if (!compareVersionsState.base || !compareVersionsState.compare) {
      message.error('Please select two versions to compare');
      return;
    }

    try {
      const result = await compareVersions(strategyId, compareVersionsState.base, compareVersionsState.compare);
      Modal.info({
        title: 'Version Comparison',
        width: 800,
        content: (
          <div>
            <Row gutter={16}>
              <Col span={8}>
                <Text strong>Files Changed:</Text> {result?.summary.filesChanged}
              </Col>
              <Col span={8}>
                <Text strong>Lines Added:</Text> <Text style={{ color: 'green' }}>+{result?.summary.linesAdded}</Text>
              </Col>
              <Col span={8}>
                <Text strong>Lines Removed:</Text> <Text style={{ color: 'red' }}>-{result?.summary.linesRemoved}</Text>
              </Col>
            </Row>
            <Divider />
            {result?.changes.map((change, index) => (
              <div key={index} style={{ marginBottom: 8 }}>
                <Tag color={change.changeType === 'added' ? 'green' : 
                            change.changeType === 'deleted' ? 'red' : 'blue'}>
                  {change.changeType}
                </Tag>
                <Text code>{change.filePath}</Text>
              </div>
            ))}
          </div>
        )
      });
    } catch (error) {
      message.error(`Failed to compare versions: ${error}`);
    }
  };

  // Handle branch switch
  const handleBranchSwitch = async (branchName: string) => {
    try {
      await switchBranch(strategyId, branchName);
      onBranchChange?.(branchName);
      message.success(`Switched to branch ${branchName}`);
    } catch (error) {
      message.error(`Failed to switch branch: ${error}`);
    }
  };

  // Version columns
  const versionColumns = [
    {
      title: 'Version',
      dataIndex: 'version',
      key: 'version',
      render: (version: string, record: any) => (
        <Button
          type="link"
          onClick={() => {
            setSelectedVersion(record.versionId);
            setVersionDetailsVisible(true);
          }}
        >
          {version}
        </Button>
      )
    },
    {
      title: 'Branch',
      dataIndex: 'branch',
      key: 'branch',
      render: (branch: string) => (
        <Tag color="blue" icon={<GitlabOutlined />}>
          {branch}
        </Tag>
      )
    },
    {
      title: 'Commit',
      dataIndex: 'commit',
      key: 'commit',
      render: (commit: string) => (
        <Text code>{commit.slice(0, 8)}</Text>
      )
    },
    {
      title: 'Message',
      dataIndex: 'message',
      key: 'message',
      render: (message: string) => (
        <Tooltip title={message}>
          <Text ellipsis style={{ maxWidth: 200 }}>
            {message}
          </Text>
        </Tooltip>
      )
    },
    {
      title: 'Author',
      dataIndex: 'author',
      key: 'author',
      render: (author: string) => (
        <Space>
          <Avatar size="small" icon={<UserOutlined />} />
          {author}
        </Space>
      )
    },
    {
      title: 'Tags',
      key: 'tags',
      render: (_: any, record: any) => (
        <Space>
          {record.tags.map((tag: string) => (
            <Tag key={tag} color="purple" icon={<TagOutlined />}>
              {tag}
            </Tag>
          ))}
        </Space>
      )
    },
    {
      title: 'Created',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: Date) => new Date(timestamp).toLocaleString()
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Tooltip title="View Details">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedVersion(record.versionId);
                setVersionDetailsVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="Create Tag">
            <Button
              size="small"
              icon={<TagOutlined />}
              onClick={() => {
                tagForm.setFieldsValue({ versionId: record.versionId });
                setTagModalVisible(true);
              }}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  // Branch columns
  const branchColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: any) => (
        <Space>
          <Tag color={record.isActive ? 'green' : 'default'} icon={<GitlabOutlined />}>
            {name}
          </Tag>
          {record.isProtected && <Tag color="red">Protected</Tag>}
        </Space>
      )
    },
    {
      title: 'Head Version',
      dataIndex: 'headVersion',
      key: 'headVersion',
      render: (version: string) => <Text code>{version}</Text>
    },
    {
      title: 'Created By',
      dataIndex: 'createdBy',
      key: 'createdBy',
      render: (user: string) => (
        <Space>
          <Avatar size="small" icon={<UserOutlined />} />
          {user}
        </Space>
      )
    },
    {
      title: 'Created At',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date: Date) => new Date(date).toLocaleString()
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      render: (desc: string) => desc || <Text type="secondary">No description</Text>
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Button
            size="small"
            type={record.isActive ? 'default' : 'primary'}
            onClick={() => handleBranchSwitch(record.name)}
            disabled={record.isActive}
          >
            {record.isActive ? 'Current' : 'Switch'}
          </Button>
          <Button
            size="small"
            icon={<MergeOutlined />}
            onClick={() => {
              mergeForm.setFieldsValue({ sourceBranch: record.name });
              setMergeRequestModalVisible(true);
            }}
          >
            Merge
          </Button>
        </Space>
      )
    }
  ];

  // Merge request columns
  const mergeRequestColumns = [
    {
      title: 'Title',
      dataIndex: 'title',
      key: 'title',
      render: (title: string, record: any) => (
        <Button
          type="link"
          onClick={() => {
            setSelectedMergeRequest(record.mergeId);
          }}
        >
          {title}
        </Button>
      )
    },
    {
      title: 'Source → Target',
      key: 'branches',
      render: (_: any, record: any) => (
        <Space>
          <Tag color="blue">{record.sourceBranch}</Tag>
          →
          <Tag color="green">{record.targetBranch}</Tag>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const color = status === 'merged' ? 'green' : 
                     status === 'open' ? 'blue' :
                     status === 'conflict' ? 'red' : 'gray';
        return <Tag color={color}>{status}</Tag>;
      }
    },
    {
      title: 'Approvals',
      dataIndex: 'approvals',
      key: 'approvals',
      render: (approvals: any[], record: any) => (
        <Space>
          <Text>{approvals.length}/{record.reviewers.length}</Text>
          <Progress
            percent={(approvals.length / record.reviewers.length) * 100}
            size="small"
            showInfo={false}
          />
        </Space>
      )
    },
    {
      title: 'Requested By',
      dataIndex: 'requestedBy',
      key: 'requestedBy',
      render: (user: string) => (
        <Space>
          <Avatar size="small" icon={<UserOutlined />} />
          {user}
        </Space>
      )
    },
    {
      title: 'Created',
      dataIndex: 'requestedAt',
      key: 'requestedAt',
      render: (date: Date) => new Date(date).toLocaleString()
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          {record.status === 'open' && (
            <>
              <Button
                size="small"
                type="primary"
                onClick={() => approveMergeRequest(record.mergeId, 'user')}
              >
                Approve
              </Button>
              <Button
                size="small"
                onClick={() => mergeBranches(record.mergeId)}
                disabled={record.approvals.length < record.reviewers.length}
              >
                Merge
              </Button>
            </>
          )}
        </Space>
      )
    }
  ];

  const selectedVersionData = selectedVersion ? getVersion(selectedVersion) : null;

  return (
    <div className="version-control-interface">
      <div style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <Title level={3}>
              <GitlabOutlined /> Version Control
            </Title>
          </Col>
          <Col>
            <Space>
              <Select
                style={{ width: 200 }}
                placeholder="Select Branch"
                value={activeBranch?.name || currentBranch}
                onChange={handleBranchSwitch}
              >
                {branches.map(branch => (
                  <Option key={branch.branchId} value={branch.name}>
                    <GitlabOutlined /> {branch.name}
                    {branch.isActive && <CheckCircleOutlined style={{ color: 'green', marginLeft: 8 }} />}
                  </Option>
                ))}
              </Select>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setNewBranchModalVisible(true)}
              >
                New Branch
              </Button>
              <Button
                icon={<CodeOutlined />}
                onClick={() => setNewVersionModalVisible(true)}
              >
                New Version
              </Button>
              <Button
                icon={<DiffOutlined />}
                onClick={() => setCompareModalVisible(true)}
              >
                Compare
              </Button>
            </Space>
          </Col>
        </Row>
      </div>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          closable
          style={{ marginBottom: 16 }}
        />
      )}

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab={`Versions (${versions.length})`} key="versions">
          <Card title="Version History">
            <Table
              dataSource={versions}
              columns={versionColumns}
              loading={loading}
              rowKey="versionId"
              pagination={{ pageSize: 20 }}
            />
          </Card>
        </TabPane>

        <TabPane tab={`Branches (${branches.length})`} key="branches">
          <Card 
            title="Branch Management"
            extra={
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setNewBranchModalVisible(true)}
              >
                Create Branch
              </Button>
            }
          >
            <Table
              dataSource={branches}
              columns={branchColumns}
              loading={loading}
              rowKey="branchId"
              pagination={false}
            />
          </Card>
        </TabPane>

        <TabPane tab={`Merge Requests (${mergeRequests.length})`} key="mergeRequests">
          <Card 
            title="Merge Requests"
            extra={
              <Button
                type="primary"
                icon={<MergeOutlined />}
                onClick={() => setMergeRequestModalVisible(true)}
              >
                Create Merge Request
              </Button>
            }
          >
            <Table
              dataSource={mergeRequests}
              columns={mergeRequestColumns}
              loading={loading}
              rowKey="mergeId"
              pagination={{ pageSize: 20 }}
            />
          </Card>
        </TabPane>

        <TabPane tab={`Tags (${tags.length})`} key="tags">
          <Card 
            title="Tags & Releases"
            extra={
              <Button
                type="primary"
                icon={<TagOutlined />}
                onClick={() => setTagModalVisible(true)}
              >
                Create Tag
              </Button>
            }
          >
            <List
              dataSource={tags}
              renderItem={tag => (
                <List.Item
                  actions={[
                    <Button key="edit" size="small" icon={<EditOutlined />} />,
                    <Button key="delete" size="small" danger icon={<DeleteOutlined />} />
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      <Avatar
                        style={{ backgroundColor: tag.isRelease ? '#52c41a' : '#1890ff' }}
                        icon={<TagOutlined />}
                      />
                    }
                    title={
                      <Space>
                        <Text strong>{tag.name}</Text>
                        {tag.isRelease && <Tag color="green">Release</Tag>}
                      </Space>
                    }
                    description={
                      <div>
                        <Paragraph>{tag.description}</Paragraph>
                        <Text type="secondary">
                          Created by {tag.taggedBy} on {new Date(tag.taggedAt).toLocaleString()}
                        </Text>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* New Branch Modal */}
      <Modal
        title="Create New Branch"
        open={newBranchModalVisible}
        onCancel={() => setNewBranchModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={branchForm}
          layout="vertical"
          onFinish={handleCreateBranch}
        >
          <Form.Item
            name="name"
            label="Branch Name"
            rules={[{ required: true, message: 'Please enter branch name' }]}
          >
            <Input placeholder="feature/new-strategy" />
          </Form.Item>

          <Form.Item
            name="baseVersion"
            label="Base Version"
          >
            <Select placeholder="Select base version (optional)">
              {versions.map(version => (
                <Option key={version.versionId} value={version.version}>
                  {version.version} - {version.message}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="description"
            label="Description"
          >
            <TextArea rows={3} placeholder="Describe the purpose of this branch" />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                Create Branch
              </Button>
              <Button onClick={() => setNewBranchModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* New Version Modal */}
      <Modal
        title="Create New Version"
        open={newVersionModalVisible}
        onCancel={() => setNewVersionModalVisible(false)}
        footer={null}
        width={800}
      >
        <Form
          form={versionForm}
          layout="vertical"
          onFinish={handleCreateVersion}
        >
          <Form.Item
            name="message"
            label="Commit Message"
            rules={[{ required: true, message: 'Please enter commit message' }]}
          >
            <Input placeholder="Add new trading signal logic" />
          </Form.Item>

          <Form.Item
            name="branch"
            label="Branch"
          >
            <Select defaultValue={currentBranch} disabled>
              <Option value={currentBranch}>{currentBranch}</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="strategyCode"
            label="Strategy Code"
            rules={[{ required: true, message: 'Please enter strategy code' }]}
          >
            <TextArea
              rows={10}
              placeholder="# Strategy implementation code"
              style={{ fontFamily: 'monospace' }}
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                Create Version
              </Button>
              <Button onClick={() => setNewVersionModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Merge Request Modal */}
      <Modal
        title="Create Merge Request"
        open={mergeRequestModalVisible}
        onCancel={() => setMergeRequestModalVisible(false)}
        footer={null}
        width={700}
      >
        <Form
          form={mergeForm}
          layout="vertical"
          onFinish={handleCreateMergeRequest}
        >
          <Form.Item
            name="title"
            label="Title"
            rules={[{ required: true, message: 'Please enter merge request title' }]}
          >
            <Input placeholder="Merge feature/new-strategy into main" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="sourceBranch"
                label="Source Branch"
                rules={[{ required: true, message: 'Please select source branch' }]}
              >
                <Select placeholder="Select source branch">
                  {branches.map(branch => (
                    <Option key={branch.branchId} value={branch.name}>
                      {branch.name}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="targetBranch"
                label="Target Branch"
                rules={[{ required: true, message: 'Please select target branch' }]}
              >
                <Select placeholder="Select target branch">
                  {branches.map(branch => (
                    <Option key={branch.branchId} value={branch.name}>
                      {branch.name}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="description"
            label="Description"
          >
            <TextArea
              rows={4}
              placeholder="Describe the changes and rationale for this merge"
            />
          </Form.Item>

          <Form.Item
            name="reviewers"
            label="Reviewers"
          >
            <Select mode="multiple" placeholder="Select reviewers">
              <Option value="reviewer1">Reviewer 1</Option>
              <Option value="reviewer2">Reviewer 2</Option>
              <Option value="reviewer3">Reviewer 3</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                Create Merge Request
              </Button>
              <Button onClick={() => setMergeRequestModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Tag Modal */}
      <Modal
        title="Create Tag"
        open={tagModalVisible}
        onCancel={() => setTagModalVisible(false)}
        footer={null}
      >
        <Form
          form={tagForm}
          layout="vertical"
          onFinish={handleCreateTag}
        >
          <Form.Item
            name="versionId"
            label="Version"
            rules={[{ required: true, message: 'Please select version' }]}
          >
            <Select placeholder="Select version to tag">
              {versions.map(version => (
                <Option key={version.versionId} value={version.versionId}>
                  {version.version} - {version.message}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="name"
            label="Tag Name"
            rules={[{ required: true, message: 'Please enter tag name' }]}
          >
            <Input placeholder="v1.0.0" />
          </Form.Item>

          <Form.Item
            name="description"
            label="Description"
          >
            <TextArea rows={3} placeholder="Release notes or tag description" />
          </Form.Item>

          <Form.Item
            name="isRelease"
            valuePropName="checked"
          >
            <Space>
              <input type="checkbox" />
              <Text>Mark as release</Text>
            </Space>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                Create Tag
              </Button>
              <Button onClick={() => setTagModalVisible(false)}>
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Compare Modal */}
      <Modal
        title="Compare Versions"
        open={compareModalVisible}
        onCancel={() => setCompareModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setCompareModalVisible(false)}>
            Cancel
          </Button>,
          <Button key="compare" type="primary" onClick={handleCompareVersions}>
            Compare
          </Button>
        ]}
      >
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item label="Base Version">
              <Select
                placeholder="Select base version"
                value={compareVersionsState.base}
                onChange={(value) => setCompareVersionsState(prev => ({ ...prev, base: value }))}
              >
                {versions.map(version => (
                  <Option key={version.versionId} value={version.version}>
                    {version.version} - {version.message}
                  </Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item label="Compare Version">
              <Select
                placeholder="Select compare version"
                value={compareVersionsState.compare}
                onChange={(value) => setCompareVersionsState(prev => ({ ...prev, compare: value }))}
              >
                {versions.map(version => (
                  <Option key={version.versionId} value={version.version}>
                    {version.version} - {version.message}
                  </Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
        </Row>
      </Modal>

      {/* Version Details Drawer */}
      <Drawer
        title="Version Details"
        placement="right"
        width={800}
        open={versionDetailsVisible}
        onClose={() => setVersionDetailsVisible(false)}
      >
        {selectedVersionData && (
          <div>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Card size="small" title="Version Information">
                <Row gutter={16}>
                  <Col span={8}>
                    <Text strong>Version:</Text><br />
                    <Tag color="blue">{selectedVersionData.version}</Tag>
                  </Col>
                  <Col span={8}>
                    <Text strong>Branch:</Text><br />
                    <Tag color="green">{selectedVersionData.branch}</Tag>
                  </Col>
                  <Col span={8}>
                    <Text strong>Commit:</Text><br />
                    <Text code>{selectedVersionData.commit}</Text>
                  </Col>
                </Row>
                <Divider />
                <Paragraph>
                  <Text strong>Message:</Text><br />
                  {selectedVersionData.message}
                </Paragraph>
                <Text type="secondary">
                  Created by {selectedVersionData.author} on{' '}
                  {new Date(selectedVersionData.timestamp).toLocaleString()}
                </Text>
              </Card>

              <Card size="small" title="Strategy Code">
                <pre style={{ 
                  backgroundColor: '#f6f8fa', 
                  padding: 16, 
                  borderRadius: 6,
                  overflow: 'auto',
                  maxHeight: 400
                }}>
                  <code>{selectedVersionData.strategyCode}</code>
                </pre>
              </Card>

              <Card size="small" title="Configuration">
                <pre style={{ 
                  backgroundColor: '#f6f8fa', 
                  padding: 16, 
                  borderRadius: 6,
                  overflow: 'auto'
                }}>
                  <code>{JSON.stringify(selectedVersionData.strategyConfig, null, 2)}</code>
                </pre>
              </Card>

              {selectedVersionData.tags.length > 0 && (
                <Card size="small" title="Tags">
                  <Space>
                    {selectedVersionData.tags.map(tag => (
                      <Tag key={tag} color="purple" icon={<TagOutlined />}>
                        {tag}
                      </Tag>
                    ))}
                  </Space>
                </Card>
              )}
            </Space>
          </div>
        )}
      </Drawer>
    </div>
  );
};