import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Tree,
  Button,
  Table,
  Space,
  Typography,
  Row,
  Col,
  Tag,
  Timeline,
  Modal,
  Form,
  Input,
  Select,
  Alert,
  Tooltip,
  Badge,
  Divider,
  List,
  Avatar,
  notification,
  Popconfirm,
  Tabs,
  Statistic
} from 'antd';
import {
  BranchesOutlined,
  TagsOutlined,
  CodeOutlined,
  MergeOutlined,
  RollbackOutlined,
  CopyOutlined,
  ForkOutlined,
  UserOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  FileTextOutlined,
  DiffOutlined,
  HistoryOutlined,
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  EyeOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';
import type { ColumnType } from 'antd/es/table';
import type { DataNode } from 'antd/es/tree';
import type {
  StrategyVersionControlProps,
  GitLikeVersion,
  VersionDiff,
  ParameterChange,
  ConfigurationChange,
  VersionControlRequest
} from './types/deploymentTypes';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

interface Branch {
  name: string;
  commit_hash: string;
  last_commit: Date;
  author: string;
  is_default: boolean;
  version_count: number;
}

interface GitTag {
  name: string;
  commit_hash: string;
  created_at: Date;
  author: string;
  message?: string;
}

const StrategyVersionControl: React.FC<StrategyVersionControlProps> = ({
  strategyId,
  showDiffs = true,
  allowBranching = true,
  onVersionChange
}) => {
  const [form] = Form.useForm();
  const [versions, setVersions] = useState<GitLikeVersion[]>([]);
  const [branches, setBranches] = useState<Branch[]>([]);
  const [tags, setTags] = useState<GitTag[]>([]);
  const [currentBranch, setCurrentBranch] = useState<string>('main');
  const [selectedVersion, setSelectedVersion] = useState<GitLikeVersion | null>(null);
  const [compareVersion, setCompareVersion] = useState<GitLikeVersion | null>(null);
  const [versionDiff, setVersionDiff] = useState<VersionDiff | null>(null);
  const [loading, setLoading] = useState(false);
  const [showCommitModal, setShowCommitModal] = useState(false);
  const [showBranchModal, setShowBranchModal] = useState(false);
  const [showTagModal, setShowTagModal] = useState(false);
  const [showMergeModal, setShowMergeModal] = useState(false);
  const [treeData, setTreeData] = useState<DataNode[]>([]);

  useEffect(() => {
    loadVersionHistory();
    loadBranches();
    loadTags();
  }, [strategyId, currentBranch]);

  useEffect(() => {
    buildVersionTree();
  }, [versions, branches]);

  const loadVersionHistory = useCallback(async () => {
    if (!strategyId) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/versions/${strategyId}?branch=${currentBranch}`);
      if (!response.ok) throw new Error('Failed to load version history');
      
      const data = await response.json();
      setVersions(data.versions || []);
    } catch (error) {
      console.error('Error loading version history:', error);
      notification.error({
        message: 'Failed to Load Versions',
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setLoading(false);
    }
  }, [strategyId, currentBranch]);

  const loadBranches = useCallback(async () => {
    if (!strategyId) return;
    
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/versions/${strategyId}/branches`);
      if (response.ok) {
        const data = await response.json();
        setBranches(data.branches || []);
      }
    } catch (error) {
      console.error('Error loading branches:', error);
    }
  }, [strategyId]);

  const loadTags = useCallback(async () => {
    if (!strategyId) return;
    
    try {
      const response = await fetch(`${API_BASE}/api/v1/strategies/versions/${strategyId}/tags`);
      if (response.ok) {
        const data = await response.json();
        setTags(data.tags || []);
      }
    } catch (error) {
      console.error('Error loading tags:', error);
    }
  }, [strategyId]);

  const buildVersionTree = () => {
    if (versions.length === 0) return;

    const treeNodes: DataNode[] = branches.map(branch => ({
      title: (
        <div className="flex items-center justify-between">
          <Space>
            <BranchesOutlined />
            <Text strong>{branch.name}</Text>
            {branch.is_default && <Tag color="blue">default</Tag>}
          </Space>
          <Badge count={branch.version_count} showZero style={{ backgroundColor: '#52c41a' }} />
        </div>
      ),
      key: `branch_${branch.name}`,
      selectable: false,
      children: versions
        .filter(v => v.branch === branch.name)
        .slice(0, 10) // Show last 10 commits per branch
        .map(version => ({
          title: (
            <div className="flex items-center justify-between w-full">
              <Space size="small">
                <CodeOutlined style={{ color: '#1890ff' }} />
                <Text code style={{ fontSize: '12px' }}>{version.commit_hash.substring(0, 8)}</Text>
                <Text style={{ fontSize: '12px' }}>{version.commit_message}</Text>
              </Space>
              <Space size="small">
                {version.tags.map(tag => (
                  <Tag key={tag} color="orange" size="small">{tag}</Tag>
                ))}
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  {dayjs(version.timestamp).fromNow()}
                </Text>
              </Space>
            </div>
          ),
          key: version.version_id,
          isLeaf: true
        }))
    }));

    setTreeData(treeNodes);
  };

  const performVersionOperation = async (operation: string, data: any) => {
    setLoading(true);
    try {
      const request: VersionControlRequest = {
        strategy_id: strategyId,
        operation: operation as any,
        ...data
      };
      
      const response = await fetch(`${API_BASE}/api/v1/strategies/versions/${strategyId}/operation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });
      
      if (!response.ok) throw new Error(`Failed to ${operation}`);
      
      const result = await response.json();
      
      notification.success({
        message: `${operation.charAt(0).toUpperCase() + operation.slice(1)} Successful`,
        description: result.message || `${operation} completed successfully`
      });
      
      // Reload data
      await loadVersionHistory();
      await loadBranches();
      await loadTags();
      
      return result;
    } catch (error) {
      console.error(`Error performing ${operation}:`, error);
      notification.error({
        message: `${operation.charAt(0).toUpperCase() + operation.slice(1)} Failed`,
        description: error instanceof Error ? error.message : 'Unknown error'
      });
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const commitVersion = async (values: any) => {
    try {
      await performVersionOperation('commit', {
        message: values.commit_message,
        metadata: values.metadata ? JSON.parse(values.metadata) : undefined
      });
      setShowCommitModal(false);
      form.resetFields();
    } catch (error) {
      // Error already handled in performVersionOperation
    }
  };

  const createBranch = async (values: any) => {
    try {
      await performVersionOperation('branch', {
        target: values.branch_name,
        message: values.description
      });
      setShowBranchModal(false);
      form.resetFields();
    } catch (error) {
      // Error already handled in performVersionOperation
    }
  };

  const createTag = async (values: any) => {
    try {
      await performVersionOperation('tag', {
        target: values.tag_name,
        message: values.message
      });
      setShowTagModal(false);
      form.resetFields();
    } catch (error) {
      // Error already handled in performVersionOperation
    }
  };

  const mergeBranch = async (values: any) => {
    try {
      await performVersionOperation('merge', {
        target: values.source_branch,
        message: values.merge_message
      });
      setShowMergeModal(false);
      form.resetFields();
    } catch (error) {
      // Error already handled in performVersionOperation
    }
  };

  const rollbackToVersion = async (version: GitLikeVersion) => {
    try {
      await performVersionOperation('rollback', {
        target: version.version_id,
        message: `Rollback to ${version.commit_hash.substring(0, 8)}`
      });
      onVersionChange?.(version.version_id);
    } catch (error) {
      // Error already handled in performVersionOperation
    }
  };

  const loadVersionDiff = async (version1: GitLikeVersion, version2: GitLikeVersion) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/v1/strategies/versions/${strategyId}/diff?v1=${version1.version_id}&v2=${version2.version_id}`
      );
      if (!response.ok) throw new Error('Failed to load version diff');
      
      const diff: VersionDiff = await response.json();
      setVersionDiff(diff);
    } catch (error) {
      console.error('Error loading version diff:', error);
      notification.error({
        message: 'Failed to Load Diff',
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  };

  const versionColumns: ColumnType<GitLikeVersion>[] = [
    {
      title: 'Commit',
      key: 'commit',
      width: 120,
      render: (_, version) => (
        <Space direction="vertical" size="small">
          <Text code>{version.commit_hash.substring(0, 8)}</Text>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {version.version_number}
          </Text>
        </Space>
      )
    },
    {
      title: 'Message',
      dataIndex: 'commit_message',
      key: 'commit_message',
      ellipsis: true
    },
    {
      title: 'Author',
      key: 'author',
      width: 120,
      render: (_, version) => (
        <Space>
          <Avatar size="small" icon={<UserOutlined />} />
          <Text style={{ fontSize: '12px' }}>{version.author}</Text>
        </Space>
      )
    },
    {
      title: 'Branch',
      dataIndex: 'branch',
      key: 'branch',
      width: 100,
      render: (branch: string) => <Tag color="blue">{branch}</Tag>
    },
    {
      title: 'Tags',
      dataIndex: 'tags',
      key: 'tags',
      width: 120,
      render: (tags: string[]) => (
        <Space wrap>
          {tags.map(tag => (
            <Tag key={tag} color="orange" size="small">{tag}</Tag>
          ))}
        </Space>
      )
    },
    {
      title: 'Date',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 120,
      render: (timestamp: Date) => (
        <Tooltip title={dayjs(timestamp).format('YYYY-MM-DD HH:mm:ss')}>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {dayjs(timestamp).fromNow()}
          </Text>
        </Tooltip>
      ),
      sorter: (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
      defaultSortOrder: 'descend'
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      fixed: 'right',
      render: (_, version) => (
        <Space size="small">
          <Tooltip title="View Details">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => setSelectedVersion(version)}
            />
          </Tooltip>
          
          {showDiffs && (
            <Tooltip title="Compare">
              <Button
                size="small"
                icon={<DiffOutlined />}
                onClick={() => {
                  setCompareVersion(version);
                  if (selectedVersion) {
                    loadVersionDiff(selectedVersion, version);
                  }
                }}
              />
            </Tooltip>
          )}
          
          <Popconfirm
            title="Rollback to this version?"
            description="This will create a new commit reverting to this version."
            onConfirm={() => rollbackToVersion(version)}
          >
            <Tooltip title="Rollback">
              <Button
                size="small"
                icon={<RollbackOutlined />}
                danger
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      )
    }
  ];

  const renderVersionDiff = () => {
    if (!versionDiff || !selectedVersion || !compareVersion) return null;
    
    return (
      <Card title="Version Comparison" className="mb-4">
        <Row gutter={16} className="mb-4">
          <Col span={12}>
            <Alert
              message={`Version A: ${selectedVersion.commit_hash.substring(0, 8)}`}
              description={selectedVersion.commit_message}
              type="info"
            />
          </Col>
          <Col span={12}>
            <Alert
              message={`Version B: ${compareVersion.commit_hash.substring(0, 8)}`}
              description={compareVersion.commit_message}
              type="warning"
            />
          </Col>
        </Row>
        
        <Tabs>
          <TabPane tab="File Changes" key="files">
            <List
              size="small"
              dataSource={[
                ...versionDiff.added_files.map(f => ({ file: f, type: 'added' })),
                ...versionDiff.modified_files.map(f => ({ file: f, type: 'modified' })),
                ...versionDiff.deleted_files.map(f => ({ file: f, type: 'deleted' }))
              ]}
              renderItem={({ file, type }) => (
                <List.Item>
                  <Space>
                    <Tag color={
                      type === 'added' ? 'success' :
                      type === 'modified' ? 'warning' : 'error'
                    }>
                      {type.charAt(0).toUpperCase() + type.slice(1)}
                    </Tag>
                    <Text code>{file}</Text>
                  </Space>
                </List.Item>
              )}
            />
          </TabPane>
          
          <TabPane tab="Parameter Changes" key="parameters">
            <Table
              size="small"
              dataSource={versionDiff.parameter_changes}
              columns={[
                { title: 'Parameter', dataIndex: 'parameter_name', key: 'parameter' },
                { 
                  title: 'Change Type', 
                  dataIndex: 'change_type', 
                  key: 'type',
                  render: (type: string) => (
                    <Tag color={
                      type === 'added' ? 'success' :
                      type === 'modified' ? 'warning' : 'error'
                    }>
                      {type.toUpperCase()}
                    </Tag>
                  )
                },
                { title: 'Old Value', dataIndex: 'old_value', key: 'old' },
                { title: 'New Value', dataIndex: 'new_value', key: 'new' }
              ]}
              pagination={false}
            />
          </TabPane>
          
          <TabPane tab="Configuration Changes" key="config">
            <Table
              size="small"
              dataSource={versionDiff.configuration_changes}
              columns={[
                { title: 'Section', dataIndex: 'section', key: 'section' },
                { title: 'Field', dataIndex: 'field', key: 'field' },
                { 
                  title: 'Impact', 
                  dataIndex: 'impact_level', 
                  key: 'impact',
                  render: (level: string) => (
                    <Tag color={
                      level === 'critical' ? 'red' :
                      level === 'high' ? 'orange' :
                      level === 'medium' ? 'yellow' : 'green'
                    }>
                      {level.toUpperCase()}
                    </Tag>
                  )
                },
                { title: 'Old Value', dataIndex: 'old_value', key: 'old' },
                { title: 'New Value', dataIndex: 'new_value', key: 'new' }
              ]}
              pagination={false}
            />
          </TabPane>
        </Tabs>
      </Card>
    );
  };

  return (
    <div className="strategy-version-control">
      <Card
        title={
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <BranchesOutlined />
              <span>Version Control</span>
              <Select
                value={currentBranch}
                onChange={setCurrentBranch}
                style={{ width: 120 }}
                size="small"
              >
                {branches.map(branch => (
                  <Option key={branch.name} value={branch.name}>
                    {branch.name}
                  </Option>
                ))}
              </Select>
            </div>
            <Space>
              <Button
                icon={<CodeOutlined />}
                onClick={() => setShowCommitModal(true)}
              >
                Commit
              </Button>
              
              {allowBranching && (
                <>
                  <Button
                    icon={<BranchesOutlined />}
                    onClick={() => setShowBranchModal(true)}
                  >
                    Branch
                  </Button>
                  
                  <Button
                    icon={<MergeOutlined />}
                    onClick={() => setShowMergeModal(true)}
                    disabled={branches.length <= 1}
                  >
                    Merge
                  </Button>
                </>
              )}
              
              <Button
                icon={<TagsOutlined />}
                onClick={() => setShowTagModal(true)}
              >
                Tag
              </Button>
            </Space>
          </div>
        }
      >
        <Row gutter={16}>
          <Col span={8}>
            <Card title="Branch Tree" size="small" style={{ height: '600px', overflow: 'auto' }}>
              <Tree
                treeData={treeData}
                defaultExpandAll
                onSelect={(keys) => {
                  const versionId = keys[0] as string;
                  const version = versions.find(v => v.version_id === versionId);
                  if (version) {
                    setSelectedVersion(version);
                    onVersionChange?.(versionId);
                  }
                }}
              />
            </Card>
          </Col>
          
          <Col span={16}>
            <Tabs defaultActiveKey="history">
              <TabPane tab="Version History" key="history">
                <Table
                  columns={versionColumns}
                  dataSource={versions}
                  rowKey="version_id"
                  size="small"
                  scroll={{ x: 800 }}
                  pagination={{ pageSize: 20 }}
                />
              </TabPane>
              
              <TabPane tab="Branches" key="branches">
                <List
                  dataSource={branches}
                  renderItem={branch => (
                    <List.Item
                      actions={[
                        <Button
                          size="small"
                          onClick={() => setCurrentBranch(branch.name)}
                        >
                          Switch
                        </Button>
                      ]}
                    >
                      <List.Item.Meta
                        avatar={<BranchesOutlined />}
                        title={
                          <Space>
                            {branch.name}
                            {branch.is_default && <Tag color="blue">default</Tag>}
                          </Space>
                        }
                        description={
                          <Space direction="vertical" size="small">
                            <Text type="secondary">
                              Last commit: {dayjs(branch.last_commit).fromNow()} by {branch.author}
                            </Text>
                            <Badge count={branch.version_count} showZero />
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />
              </TabPane>
              
              <TabPane tab="Tags" key="tags">
                <List
                  dataSource={tags}
                  renderItem={tag => (
                    <List.Item>
                      <List.Item.Meta
                        avatar={<TagsOutlined />}
                        title={tag.name}
                        description={
                          <Space direction="vertical" size="small">
                            <Text type="secondary">
                              {tag.message || 'No description'}
                            </Text>
                            <Text type="secondary" style={{ fontSize: '11px' }}>
                              Created {dayjs(tag.created_at).fromNow()} by {tag.author}
                            </Text>
                            <Text code style={{ fontSize: '11px' }}>
                              {tag.commit_hash.substring(0, 8)}
                            </Text>
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />
              </TabPane>
            </Tabs>
          </Col>
        </Row>
        
        {renderVersionDiff()}
      </Card>

      {/* Commit Modal */}
      <Modal
        title="Create Commit"
        open={showCommitModal}
        onCancel={() => setShowCommitModal(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={commitVersion}>
          <Form.Item
            label="Commit Message"
            name="commit_message"
            rules={[{ required: true, message: 'Please enter a commit message' }]}
          >
            <TextArea rows={3} placeholder="Describe your changes..." />
          </Form.Item>
          
          <Form.Item
            label="Metadata (JSON)"
            name="metadata"
            tooltip="Optional metadata in JSON format"
          >
            <TextArea rows={3} placeholder='{"deployment": "production", "tested": true}' />
          </Form.Item>
        </Form>
      </Modal>

      {/* Branch Modal */}
      <Modal
        title="Create Branch"
        open={showBranchModal}
        onCancel={() => setShowBranchModal(false)}
        onOk={() => form.submit()}
      >
        <Form form={form} layout="vertical" onFinish={createBranch}>
          <Form.Item
            label="Branch Name"
            name="branch_name"
            rules={[{ required: true, message: 'Please enter a branch name' }]}
          >
            <Input placeholder="feature/new-algorithm" />
          </Form.Item>
          
          <Form.Item label="Description" name="description">
            <TextArea rows={2} placeholder="Branch description..." />
          </Form.Item>
        </Form>
      </Modal>

      {/* Tag Modal */}
      <Modal
        title="Create Tag"
        open={showTagModal}
        onCancel={() => setShowTagModal(false)}
        onOk={() => form.submit()}
      >
        <Form form={form} layout="vertical" onFinish={createTag}>
          <Form.Item
            label="Tag Name"
            name="tag_name"
            rules={[{ required: true, message: 'Please enter a tag name' }]}
          >
            <Input placeholder="v1.0.0" />
          </Form.Item>
          
          <Form.Item label="Message" name="message">
            <TextArea rows={2} placeholder="Release notes..." />
          </Form.Item>
        </Form>
      </Modal>

      {/* Merge Modal */}
      <Modal
        title="Merge Branch"
        open={showMergeModal}
        onCancel={() => setShowMergeModal(false)}
        onOk={() => form.submit()}
      >
        <Form form={form} layout="vertical" onFinish={mergeBranch}>
          <Form.Item
            label="Source Branch"
            name="source_branch"
            rules={[{ required: true, message: 'Please select a source branch' }]}
          >
            <Select placeholder="Select branch to merge">
              {branches
                .filter(b => b.name !== currentBranch)
                .map(branch => (
                  <Option key={branch.name} value={branch.name}>
                    {branch.name}
                  </Option>
                ))}
            </Select>
          </Form.Item>
          
          <Form.Item
            label="Merge Message"
            name="merge_message"
            rules={[{ required: true, message: 'Please enter a merge message' }]}
          >
            <TextArea rows={2} placeholder="Merge feature branch..." />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default StrategyVersionControl;