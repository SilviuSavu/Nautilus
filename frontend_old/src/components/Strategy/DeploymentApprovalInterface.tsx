import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  Input,
  Button,
  Alert,
  Descriptions,
  Tabs,
  Table,
  Tag,
  Space,
  Typography,
  Row,
  Col,
  Statistic,
  Steps,
  Timeline,
  Modal,
  Checkbox,
  Rate,
  Progress,
  Badge,
  Divider,
  message
} from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  FileTextOutlined,
  UserOutlined,
  CalendarOutlined,
  SafetyCertificateOutlined,
  LineChartOutlined,
  WarningOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import type {
  StrategyDeployment,
  ApproveDeploymentRequest,
  ApprovalInterfaceProps,
  BacktestResults,
  RiskAssessment,
  RolloutPhase
} from '../../types/deployment';

const { Text, Title, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Step } = Steps;

const DeploymentApprovalInterface: React.FC<ApprovalInterfaceProps> = ({
  deployment,
  currentUser,
  onApprove
}) => {
  const [approvalForm] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [riskReview, setRiskReview] = useState<any>(null);
  const [approvalChecklist, setApprovalChecklist] = useState<{[key: string]: boolean}>({});
  const [overallRating, setOverallRating] = useState<number>(0);
  const [showRiskDetails, setShowRiskDetails] = useState(false);

  const userCanApprove = deployment.approvalChain.some(
    approval => approval.approverId === currentUser.id && approval.status === 'pending'
  );

  const currentApproval = deployment.approvalChain.find(
    approval => approval.approverId === currentUser.id
  );

  useEffect(() => {
    loadRiskReview();
  }, [deployment.deploymentId]);

  const loadRiskReview = async () => {
    try {
      const response = await fetch(`/api/v1/nautilus/deployment/${deployment.deploymentId}/risk-review`);
      const data = await response.json();
      setRiskReview(data);
    } catch (error) {
      console.error('Error loading risk review:', error);
    }
  };

  const handleApproval = async (approved: boolean) => {
    if (!userCanApprove) {
      message.error('You are not authorized to approve this deployment');
      return;
    }

    // Check if all required checklist items are completed for approval
    if (approved) {
      const requiredChecks = Object.keys(getApprovalChecklist());
      const completedChecks = Object.keys(approvalChecklist).filter(key => approvalChecklist[key]);
      
      if (completedChecks.length < requiredChecks.length) {
        message.error('Please complete all approval checklist items before approving');
        return;
      }

      if (overallRating === 0) {
        message.error('Please provide an overall risk rating');
        return;
      }
    }

    setLoading(true);
    try {
      const values = approvalForm.getFieldsValue();
      const request: ApproveDeploymentRequest = {
        deploymentId: deployment.deploymentId,
        comments: values.comments,
        conditionalApproval: values.conditionalApproval,
        conditions: values.conditions?.split('\n').filter((c: string) => c.trim())
      };

      await onApprove?.(approved, values.comments);
      message.success(`Deployment ${approved ? 'approved' : 'rejected'} successfully`);
    } catch (error) {
      console.error('Approval failed:', error);
      message.error('Failed to process approval');
    } finally {
      setLoading(false);
    }
  };

  const getApprovalChecklist = () => {
    const baseChecks = {
      'strategy_reviewed': 'Strategy configuration has been reviewed',
      'risk_assessment_ok': 'Risk assessment is acceptable',
      'backtest_results_adequate': 'Backtest results meet requirements',
      'rollout_plan_reasonable': 'Rollout plan is reasonable and safe'
    };

    if (currentUser.role === 'risk_manager') {
      return {
        ...baseChecks,
        'position_limits_ok': 'Position limits are within portfolio constraints',
        'var_estimate_acceptable': 'VaR estimate is within acceptable range',
        'correlation_risk_managed': 'Correlation risk is properly managed',
        'drawdown_limits_ok': 'Maximum drawdown limits are appropriate'
      };
    }

    if (currentUser.role === 'compliance') {
      return {
        ...baseChecks,
        'regulatory_compliant': 'Strategy complies with regulatory requirements',
        'documentation_complete': 'All required documentation is complete',
        'audit_trail_adequate': 'Audit trail is comprehensive'
      };
    }

    return baseChecks;
  };

  const renderDeploymentSummary = () => (
    <Card title="Deployment Summary" size="small">
      <Descriptions bordered size="small">
        <Descriptions.Item label="Strategy" span={2}>
          {deployment.strategyId}
        </Descriptions.Item>
        <Descriptions.Item label="Version">
          <Tag color="blue">{deployment.version}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="Environment" span={2}>
          {deployment.deploymentConfig.environment}
        </Descriptions.Item>
        <Descriptions.Item label="Status">
          <Tag color="orange">{deployment.status.replace('_', ' ').toUpperCase()}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="Created By" span={2}>
          <div className="flex items-center space-x-1">
            <UserOutlined />
            <span>{deployment.createdBy}</span>
          </div>
        </Descriptions.Item>
        <Descriptions.Item label="Created At">
          <div className="flex items-center space-x-1">
            <CalendarOutlined />
            <span>{deployment.createdAt.toLocaleString()}</span>
          </div>
        </Descriptions.Item>
      </Descriptions>
    </Card>
  );

  const renderRiskAssessment = () => (
    <Card title="Risk Assessment" size="small">
      {riskReview ? (
        <>
          <Row gutter={16} className="mb-4">
            <Col span={6}>
              <Statistic
                title="Risk Level"
                value={riskReview.risk_level}
                valueStyle={{ 
                  color: riskReview.risk_level === 'low' ? '#3f8600' : 
                         riskReview.risk_level === 'medium' ? '#faad14' : '#cf1322' 
                }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Portfolio Impact"
                value={riskReview.portfolioImpact}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="Max Drawdown"
                value={(riskReview.maxDrawdownEstimate * 100).toFixed(1)}
                suffix="%"
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="VaR Estimate"
                value={(riskReview.varEstimate * 100).toFixed(1)}
                suffix="%"
              />
            </Col>
          </Row>

          {riskReview.warnings?.length > 0 && (
            <Alert
              message="Risk Warnings"
              description={
                <ul className="mb-0">
                  {riskReview.warnings.map((warning: string, index: number) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              }
              type="warning"
              className="mb-4"
            />
          )}

          {riskReview.recommendations?.length > 0 && (
            <Alert
              message="Recommendations"
              description={
                <ul className="mb-0">
                  {riskReview.recommendations.map((rec: string, index: number) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              }
              type="info"
            />
          )}

          <div className="mt-4">
            <Button type="link" onClick={() => setShowRiskDetails(true)}>
              View Detailed Risk Analysis
            </Button>
          </div>
        </>
      ) : (
        <div className="text-center py-4">
          <Text type="secondary">Loading risk assessment...</Text>
        </div>
      )}
    </Card>
  );

  const renderBacktestResults = () => (
    <Card title="Backtest Performance" size="small">
      {deployment.backtestResults ? (
        <Row gutter={16}>
          <Col span={8}>
            <Statistic
              title="Total Return"
              value={(deployment.backtestResults.totalReturn * 100).toFixed(2)}
              suffix="%"
              valueStyle={{ color: deployment.backtestResults.totalReturn >= 0 ? '#3f8600' : '#cf1322' }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Sharpe Ratio"
              value={deployment.backtestResults.sharpeRatio.toFixed(2)}
              valueStyle={{ color: deployment.backtestResults.sharpeRatio >= 1 ? '#3f8600' : '#faad14' }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Max Drawdown"
              value={(deployment.backtestResults.maxDrawdown * 100).toFixed(2)}
              suffix="%"
              valueStyle={{ color: deployment.backtestResults.maxDrawdown <= 0.1 ? '#3f8600' : '#cf1322' }}
            />
          </Col>
        </Row>
      ) : (
        <Alert message="No backtest results available" type="warning" />
      )}
    </Card>
  );

  const renderRolloutPlan = () => (
    <Card title="Rollout Plan" size="small">
      <Table
        dataSource={deployment.rolloutPlan.phases}
        pagination={false}
        size="small"
        rowKey="name"
        columns={[
          {
            title: 'Phase',
            dataIndex: 'name',
            render: (name: string) => name.replace('_', ' ').toUpperCase()
          },
          {
            title: 'Position Size',
            dataIndex: 'positionSizePercent',
            render: (size: number) => `${size}%`
          },
          {
            title: 'Duration',
            dataIndex: 'duration',
            render: (duration: number) => duration === -1 ? 'Indefinite' : `${Math.round(duration / 60)} min`
          },
          {
            title: 'Success Criteria',
            key: 'criteria',
            render: (_, phase: RolloutPhase) => (
              <div>
                {phase.successCriteria.minTrades && `Min trades: ${phase.successCriteria.minTrades}`}
                {phase.successCriteria.maxDrawdown && ` | Max DD: ${(phase.successCriteria.maxDrawdown * 100).toFixed(1)}%`}
                {phase.successCriteria.ongoing && 'Ongoing monitoring'}
              </div>
            )
          }
        ]}
      />
    </Card>
  );

  const renderApprovalChecklist = () => (
    <Card title="Approval Checklist" size="small">
      <div className="space-y-3">
        {Object.entries(getApprovalChecklist()).map(([key, description]) => (
          <div key={key} className="flex items-start space-x-2">
            <Checkbox
              checked={approvalChecklist[key] || false}
              onChange={(e) => setApprovalChecklist(prev => ({
                ...prev,
                [key]: e.target.checked
              }))}
              disabled={!userCanApprove}
            />
            <Text>{description}</Text>
          </div>
        ))}
      </div>

      <Divider />

      <div>
        <Text strong>Overall Risk Rating:</Text>
        <div className="mt-2">
          <Rate
            value={overallRating}
            onChange={setOverallRating}
            disabled={!userCanApprove}
            character={<SafetyCertificateOutlined />}
          />
          <Text type="secondary" className="ml-2">
            {overallRating === 1 && 'Very High Risk'}
            {overallRating === 2 && 'High Risk'}
            {overallRating === 3 && 'Medium Risk'}
            {overallRating === 4 && 'Low Risk'}
            {overallRating === 5 && 'Very Low Risk'}
          </Text>
        </div>
      </div>
    </Card>
  );

  const renderApprovalChain = () => (
    <Card title="Approval Progress" size="small">
      <Timeline>
        {deployment.approvalChain.map((approval, index) => (
          <Timeline.Item
            key={approval.approvalId}
            color={
              approval.status === 'approved' ? 'green' :
              approval.status === 'rejected' ? 'red' :
              approval.approverId === currentUser.id ? 'blue' : 'gray'
            }
            dot={
              approval.status === 'approved' ? <CheckCircleOutlined /> :
              approval.status === 'rejected' ? <CloseCircleOutlined /> :
              approval.approverId === currentUser.id ? <ExclamationCircleOutlined /> :
              <UserOutlined />
            }
          >
            <div>
              <Text strong>
                {approval.approverName} ({approval.requiredRole})
                {approval.approverId === currentUser.id && ' (You)'}
              </Text>
              <br />
              <Badge
                status={
                  approval.status === 'approved' ? 'success' :
                  approval.status === 'rejected' ? 'error' : 'processing'
                }
                text={approval.status.toUpperCase()}
              />
              {approval.approvedAt && (
                <Text type="secondary" className="ml-2">
                  - {approval.approvedAt.toLocaleString()}
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
  );

  const renderApprovalForm = () => (
    <Card title="Approval Decision" size="small">
      {userCanApprove ? (
        <Form form={approvalForm} layout="vertical">
          <Form.Item label="Comments" name="comments">
            <Input.TextArea
              rows={4}
              placeholder="Add your review comments..."
            />
          </Form.Item>

          <Form.Item name="conditionalApproval" valuePropName="checked">
            <Checkbox>Conditional approval</Checkbox>
          </Form.Item>

          <Form.Item
            label="Conditions (one per line)"
            name="conditions"
            dependencies={['conditionalApproval']}
          >
            <Input.TextArea
              rows={3}
              placeholder="List any conditions for approval..."
            />
          </Form.Item>

          <div className="flex justify-end space-x-2">
            <Button
              danger
              onClick={() => handleApproval(false)}
              loading={loading}
            >
              Reject
            </Button>
            <Button
              type="primary"
              onClick={() => handleApproval(true)}
              loading={loading}
              disabled={Object.keys(approvalChecklist).filter(key => approvalChecklist[key]).length < Object.keys(getApprovalChecklist()).length}
            >
              Approve
            </Button>
          </div>
        </Form>
      ) : (
        <Alert
          message={
            currentApproval?.status === 'approved' ? 'You have already approved this deployment' :
            currentApproval?.status === 'rejected' ? 'You have already rejected this deployment' :
            'You are not authorized to approve this deployment'
          }
          type={currentApproval?.status === 'approved' ? 'success' : 'info'}
        />
      )}
    </Card>
  );

  return (
    <div className="deployment-approval-interface">
      <Card
        title={
          <div className="flex items-center space-x-2">
            <FileTextOutlined className="text-blue-600" />
            <span>Deployment Approval Review</span>
          </div>
        }
      >
        <Tabs defaultActiveKey="summary">
          <TabPane tab="Summary" key="summary">
            <div className="space-y-4">
              {renderDeploymentSummary()}
              {renderApprovalChain()}
            </div>
          </TabPane>

          <TabPane tab="Risk Assessment" key="risk">
            <div className="space-y-4">
              {renderRiskAssessment()}
              {renderBacktestResults()}
            </div>
          </TabPane>

          <TabPane tab="Configuration" key="config">
            <div className="space-y-4">
              {renderRolloutPlan()}
              {/* Additional configuration details can go here */}
            </div>
          </TabPane>

          <TabPane tab="Review" key="review">
            <div className="space-y-4">
              {renderApprovalChecklist()}
              {renderApprovalForm()}
            </div>
          </TabPane>
        </Tabs>
      </Card>

      {/* Risk Details Modal */}
      <Modal
        title="Detailed Risk Analysis"
        visible={showRiskDetails}
        onCancel={() => setShowRiskDetails(false)}
        footer={null}
        width={800}
      >
        {riskReview && (
          <div>
            <Descriptions bordered>
              <Descriptions.Item label="Risk Level" span={3}>
                <Tag color={riskReview.risk_level === 'low' ? 'green' : 
                           riskReview.risk_level === 'medium' ? 'orange' : 'red'}>
                  {riskReview.risk_level.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Portfolio Impact">
                {riskReview.portfolioImpact}
              </Descriptions.Item>
              <Descriptions.Item label="Correlation Risk">
                {riskReview.correlationRisk}
              </Descriptions.Item>
              <Descriptions.Item label="Liquidity Risk">
                {riskReview.liquidityRisk}
              </Descriptions.Item>
              <Descriptions.Item label="Max Drawdown Estimate">
                {(riskReview.maxDrawdownEstimate * 100).toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="VaR Estimate">
                {(riskReview.varEstimate * 100).toFixed(2)}%
              </Descriptions.Item>
            </Descriptions>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default DeploymentApprovalInterface;