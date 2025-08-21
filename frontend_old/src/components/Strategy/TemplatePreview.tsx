import React, { useState } from 'react';
import {
  Card,
  Tabs,
  Typography,
  Table,
  Tag,
  Space,
  Button,
  Collapse,
  Alert,
  Badge,
  Descriptions,
  CodeBlock,
  Empty
} from 'antd';
import {
  BookOutlined,
  ExperimentOutlined,
  SettingOutlined,
  SafetyOutlined,
  CodeOutlined,
  ExportOutlined
} from '@ant-design/icons';
import { StrategyTemplate, ParameterDefinition, ExampleConfig } from './types/strategyTypes';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Panel } = Collapse;

interface TemplatePreviewProps {
  template: StrategyTemplate;
  className?: string;
}

export const TemplatePreview: React.FC<TemplatePreviewProps> = ({
  template,
  className
}) => {
  const [activeTab, setActiveTab] = useState('overview');

  const parameterColumns = [
    {
      title: 'Parameter',
      dataIndex: 'display_name',
      key: 'display_name',
      render: (text: string, record: ParameterDefinition) => (
        <Space>
          <Text strong>{text}</Text>
          {record.required && <Tag color="red">Required</Tag>}
        </Space>
      )
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color="blue">{type}</Tag>
      )
    },
    {
      title: 'Default',
      dataIndex: 'default_value',
      key: 'default_value',
      render: (value: any) => (
        <Text code>{value !== undefined ? String(value) : 'None'}</Text>
      )
    },
    {
      title: 'Range/Options',
      key: 'constraints',
      render: (record: ParameterDefinition) => {
        const constraints = [];
        
        if (record.min_value !== undefined || record.max_value !== undefined) {
          const min = record.min_value ?? '-∞';
          const max = record.max_value ?? '∞';
          constraints.push(`Range: [${min}, ${max}]`);
        }
        
        if (record.allowed_values && record.allowed_values.length > 0) {
          constraints.push(`Options: ${record.allowed_values.join(', ')}`);
        }
        
        return constraints.length > 0 ? (
          <Text type="secondary" style={{ fontSize: 12 }}>
            {constraints.join(' | ')}
          </Text>
        ) : null;
      }
    },
    {
      title: 'Description',
      dataIndex: 'help_text',
      key: 'help_text',
      render: (text: string) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {text}
        </Text>
      ),
      width: '30%'
    }
  ];

  const riskParameterColumns = [
    ...parameterColumns.slice(0, -1),
    {
      title: 'Impact',
      key: 'impact_level',
      render: (record: any) => {
        const impact = record.impact_level;
        const colors = {
          low: 'green',
          medium: 'orange', 
          high: 'red',
          critical: 'red'
        };
        return <Tag color={colors[impact] || 'default'}>{impact?.toUpperCase()}</Tag>;
      }
    },
    parameterColumns[parameterColumns.length - 1]
  ];

  const renderExampleConfig = (example: ExampleConfig, index: number) => (
    <Panel
      key={index}
      header={
        <Space>
          <Text strong>{example.name}</Text>
          <Text type="secondary">— {example.description}</Text>
        </Space>
      }
    >
      <pre style={{
        backgroundColor: '#f6f8fa',
        padding: 12,
        borderRadius: 6,
        fontSize: 12,
        overflow: 'auto'
      }}>
        {JSON.stringify(example.parameters, null, 2)}
      </pre>
    </Panel>
  );

  const renderOverviewTab = () => (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Descriptions bordered size="small" column={2}>
        <Descriptions.Item label="Strategy Name" span={2}>
          <Text strong>{template.name}</Text>
        </Descriptions.Item>
        <Descriptions.Item label="Category">
          <Tag color="blue">{template.category.replace('_', ' ')}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="Python Class">
          <Text code>{template.python_class}</Text>
        </Descriptions.Item>
        <Descriptions.Item label="Parameters" span={1}>
          <Badge count={template.parameters.length} style={{ backgroundColor: '#52c41a' }} />
        </Descriptions.Item>
        <Descriptions.Item label="Risk Controls" span={1}>
          <Badge count={template.risk_parameters.length} style={{ backgroundColor: '#fa8c16' }} />
        </Descriptions.Item>
        <Descriptions.Item label="Example Configs" span={2}>
          <Badge count={template.example_configs.length} style={{ backgroundColor: '#722ed1' }} />
        </Descriptions.Item>
        <Descriptions.Item label="Created" span={1}>
          {new Date(template.created_at).toLocaleDateString()}
        </Descriptions.Item>
        <Descriptions.Item label="Updated" span={1}>
          {new Date(template.updated_at).toLocaleDateString()}
        </Descriptions.Item>
      </Descriptions>

      <Card size="small" title="Description">
        <Paragraph>{template.description}</Paragraph>
      </Card>

      {template.documentation_url && (
        <Alert
          type="info"
          message="External Documentation Available"
          description="Click below to view detailed documentation for this strategy template."
          action={
            <Button
              type="primary"
              icon={<ExportOutlined />}
              onClick={() => window.open(template.documentation_url, '_blank')}
            >
              View Documentation
            </Button>
          }
        />
      )}
    </Space>
  );

  const renderParametersTab = () => (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card 
        size="small" 
        title={
          <Space>
            <SettingOutlined />
            <Text>Strategy Parameters ({template.parameters.length})</Text>
          </Space>
        }
      >
        {template.parameters.length > 0 ? (
          <Table
            dataSource={template.parameters}
            columns={parameterColumns}
            pagination={false}
            size="small"
            rowKey="name"
          />
        ) : (
          <Empty description="No parameters defined" />
        )}
      </Card>

      <Card 
        size="small" 
        title={
          <Space>
            <SafetyOutlined />
            <Text>Risk Parameters ({template.risk_parameters.length})</Text>
          </Space>
        }
      >
        {template.risk_parameters.length > 0 ? (
          <Table
            dataSource={template.risk_parameters}
            columns={riskParameterColumns}
            pagination={false}
            size="small"
            rowKey="name"
          />
        ) : (
          <Empty description="No risk parameters defined" />
        )}
      </Card>
    </Space>
  );

  const renderExamplesTab = () => (
    <Card 
      title={
        <Space>
          <ExperimentOutlined />
          <Text>Example Configurations ({template.example_configs.length})</Text>
        </Space>
      }
    >
      {template.example_configs.length > 0 ? (
        <Collapse>
          {template.example_configs.map(renderExampleConfig)}
        </Collapse>
      ) : (
        <Empty 
          description="No example configurations available"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      )}
    </Card>
  );

  const renderIntegrationTab = () => (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Alert
        type="info"
        message="NautilusTrader Integration"
        description="This strategy template integrates with the NautilusTrader Python framework."
      />

      <Card 
        title={
          <Space>
            <CodeOutlined />
            <Text>Python Integration Details</Text>
          </Space>
        }
        size="small"
      >
        <Descriptions bordered size="small">
          <Descriptions.Item label="Strategy Class" span={3}>
            <Text code>{template.python_class}</Text>
          </Descriptions.Item>
          <Descriptions.Item label="Framework" span={3}>
            <Text>NautilusTrader</Text>
          </Descriptions.Item>
          <Descriptions.Item label="Parameter Mapping" span={3}>
            <Text type="secondary">
              Frontend parameters are automatically mapped to Python strategy constructor arguments
            </Text>
          </Descriptions.Item>
        </Descriptions>

        <div style={{ marginTop: 16 }}>
          <Text strong>Expected Python Constructor:</Text>
          <pre style={{
            backgroundColor: '#f6f8fa',
            padding: 12,
            borderRadius: 6,
            fontSize: 12,
            marginTop: 8,
            overflow: 'auto'
          }}>
{`class ${template.python_class}(Strategy):
    def __init__(self, ${template.parameters.map(p => `${p.name}: ${getTypeHint(p.type)}`).join(', ')}):
        # Strategy implementation
        pass`}
          </pre>
        </div>
      </Card>
    </Space>
  );

  const getTypeHint = (paramType: string): string => {
    switch (paramType) {
      case 'string': return 'str';
      case 'integer': return 'int';
      case 'decimal': return 'Decimal';
      case 'boolean': return 'bool';
      case 'instrument_id': return 'InstrumentId';
      case 'timeframe': return 'str';
      case 'currency': return 'str';
      case 'percentage': return 'float';
      default: return 'Any';
    }
  };

  return (
    <div className={`template-preview ${className || ''}`}>
      <Card
        title={
          <Space>
            <BookOutlined />
            <Title level={4} style={{ margin: 0 }}>
              {template.name} - Template Preview
            </Title>
          </Space>
        }
        style={{ height: '100%' }}
      >
        <Tabs activeKey={activeTab} onChange={setActiveTab} type="card">
          <TabPane
            tab={
              <Space>
                <BookOutlined />
                <span>Overview</span>
              </Space>
            }
            key="overview"
          >
            {renderOverviewTab()}
          </TabPane>

          <TabPane
            tab={
              <Space>
                <SettingOutlined />
                <span>Parameters</span>
                <Badge 
                  count={template.parameters.length + template.risk_parameters.length}
                  size="small"
                />
              </Space>
            }
            key="parameters"
          >
            {renderParametersTab()}
          </TabPane>

          <TabPane
            tab={
              <Space>
                <ExperimentOutlined />
                <span>Examples</span>
                <Badge count={template.example_configs.length} size="small" />
              </Space>
            }
            key="examples"
          >
            {renderExamplesTab()}
          </TabPane>

          <TabPane
            tab={
              <Space>
                <CodeOutlined />
                <span>Integration</span>
              </Space>
            }
            key="integration"
          >
            {renderIntegrationTab()}
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};