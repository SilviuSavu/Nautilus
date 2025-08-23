/**
 * Story 5.3: API Mapping Configuration Component
 * Data mapping configuration interface for API integrations
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Modal,
  Form,
  Input,
  Select,
  Typography,
  Tag,
  Tooltip,
  Alert,
  Row,
  Col,
  notification
} from 'antd';
import {
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  CopyOutlined,
  ExperimentOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { FieldMapping } from '../../types/export';

const { Text } = Typography;
const { Option } = Select;

interface ApiMappingConfigProps {
  mappings: FieldMapping[];
  onMappingsChange: (mappings: FieldMapping[]) => void;
  sourceFields: string[];
  targetFields: { value: string; label: string; type?: string }[];
  className?: string;
}

interface TransformationTest {
  input: any;
  output: any;
  success: boolean;
  error?: string;
}

export const ApiMappingConfig: React.FC<ApiMappingConfigProps> = ({
  mappings,
  onMappingsChange,
  sourceFields,
  targetFields,
  className
}) => {
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [testModalVisible, setTestModalVisible] = useState(false);
  const [editingMapping, setEditingMapping] = useState<FieldMapping | null>(null);
  const [editingIndex, setEditingIndex] = useState<number>(-1);
  const [form] = Form.useForm();
  const [testResults, setTestResults] = useState<TransformationTest[]>([]);

  // Common transformation templates
  const transformationTemplates = [
    { label: 'Multiply by 100 (percentage)', value: '* 100' },
    { label: 'Divide by 100', value: '/ 100' },
    { label: 'Round to 2 decimals', value: 'Math.round(value * 100) / 100' },
    { label: 'Convert to string', value: 'String(value)' },
    { label: 'Convert to integer', value: 'parseInt(value)' },
    { label: 'Absolute value', value: 'Math.abs(value)' },
    { label: 'Add 1000', value: '+ 1000' },
    { label: 'Negate value', value: '* -1' },
    { label: 'Convert timestamp', value: 'new Date(value).toISOString()' }
  ];

  // Field type information for better mapping suggestions
  const getFieldType = (fieldName: string): string => {
    const numericFields = ['price', 'quantity', 'value', 'pnl', 'commission', 'win_rate', 'sharpe_ratio'];
    const dateFields = ['timestamp', 'created_at', 'updated_at'];
    const stringFields = ['symbol', 'side', 'strategy', 'account', 'venue'];
    
    if (numericFields.some(f => fieldName.toLowerCase().includes(f))) return 'numeric';
    if (dateFields.some(f => fieldName.toLowerCase().includes(f))) return 'date';
    if (stringFields.some(f => fieldName.toLowerCase().includes(f))) return 'string';
    
    return 'unknown';
  };

  const addMapping = () => {
    setEditingMapping({
      source_field: '',
      target_field: '',
      transformation: undefined
    });
    setEditingIndex(-1);
    setEditModalVisible(true);
  };

  const editMapping = (mapping: FieldMapping, index: number) => {
    setEditingMapping(mapping);
    setEditingIndex(index);
    setEditModalVisible(true);
  };

  const deleteMapping = (index: number) => {
    const newMappings = mappings.filter((_, i) => i !== index);
    onMappingsChange(newMappings);
    notification.success({
      message: 'Mapping Deleted',
      description: 'Field mapping has been removed successfully',
    });
  };

  const duplicateMapping = (mapping: FieldMapping) => {
    const newMapping = { ...mapping, target_field: `${mapping.target_field}_copy` };
    onMappingsChange([...mappings, newMapping]);
    notification.success({
      message: 'Mapping Duplicated',
      description: 'Field mapping has been duplicated successfully',
    });
  };

  const saveMapping = async () => {
    try {
      const values = await form.validateFields();
      const newMapping: FieldMapping = {
        source_field: values.source_field,
        target_field: values.target_field,
        transformation: values.transformation || undefined
      };

      let newMappings;
      if (editingIndex >= 0) {
        // Edit existing mapping
        newMappings = [...mappings];
        newMappings[editingIndex] = newMapping;
      } else {
        // Add new mapping
        newMappings = [...mappings, newMapping];
      }

      onMappingsChange(newMappings);
      setEditModalVisible(false);
      
      notification.success({
        message: 'Mapping Saved',
        description: 'Field mapping has been saved successfully',
      });
    } catch (error) {
      // Form validation failed
    }
  };

  const testTransformation = (mapping: FieldMapping) => {
    setEditingMapping(mapping);
    setTestModalVisible(true);
    
    // Generate test data
    const testValues = generateTestData(mapping.source_field);
    const results = testValues.map(value => {
      try {
        const output = applyTransformation(value, mapping.transformation);
        return { input: value, output, success: true };
      } catch (error: any) {
        return { input: value, output: null, success: false, error: error.message };
      }
    });
    
    setTestResults(results);
  };

  const generateTestData = (sourceField: string): any[] => {
    const fieldType = getFieldType(sourceField);
    
    switch (fieldType) {
      case 'numeric':
        return [100, 0, -50, 123.456, 0.001];
      case 'date':
        return [
          '2024-01-15T10:30:00Z',
          new Date().toISOString(),
          '2023-12-31T23:59:59Z'
        ];
      case 'string':
        return ['AAPL', 'TEST', 'momentum_1', '', 'long_string_value'];
      default:
        return [100, 'test', true, null, { nested: 'object' }];
    }
  };

  const applyTransformation = (value: any, transformation?: string): any => {
    if (!transformation) return value;
    
    try {
      // Simple transformation evaluation (in production, use a safer approach)
      const code = transformation.includes('value') 
        ? transformation.replace(/value/g, JSON.stringify(value))
        : `(${value}) ${transformation}`;
      
      return Function(`"use strict"; return (${code})`)();
    } catch (error) {
      throw new Error(`Transformation failed: ${error}`);
    }
  };

  // const getSuggestions = (sourceField: string, targetField: string): string[] => {
  //   const suggestions = [];
  //   const sourceType = getFieldType(sourceField);
  //   const targetType = getFieldType(targetField);
    
  //   // Type-based suggestions
  //   if (sourceType === 'numeric' && targetField.includes('percentage')) {
  //     suggestions.push('* 100');
  //   }
    
  //   if (sourceField.includes('pnl') && targetField.includes('profit')) {
  //     suggestions.push('Math.abs(value)');
  //   }
    
  //   if (sourceField === 'timestamp' && targetType === 'date') {
  //     suggestions.push('new Date(value).toISOString()');
  //   }
    
  //   return suggestions;
  // };

  const columns = [
    {
      title: 'Source Field',
      dataIndex: 'source_field',
      key: 'source_field',
      render: (field: string) => (
        <div>
          <Text strong>{field}</Text>
          <br />
          <Tag color="blue">{getFieldType(field)}</Tag>
        </div>
      )
    },
    {
      title: 'Target Field',
      dataIndex: 'target_field',
      key: 'target_field',
      render: (field: string) => {
        const targetInfo = targetFields.find(f => f.value === field);
        return (
          <div>
            <Text strong>{targetInfo?.label || field}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>{field}</Text>
          </div>
        );
      }
    },
    {
      title: 'Transformation',
      dataIndex: 'transformation',
      key: 'transformation',
      render: (transformation: string | undefined, record: FieldMapping) => (
        <div>
          {transformation ? (
            <div>
              <Text code style={{ fontSize: 11 }}>{transformation}</Text>
              <br />
              <Button 
                type="link" 
                size="small" 
                icon={<ExperimentOutlined />}
                onClick={() => testTransformation(record)}
              >
                Test
              </Button>
            </div>
          ) : (
            <Text type="secondary">None</Text>
          )}
        </div>
      )
    },
    {
      title: 'Status',
      key: 'status',
      render: (record: FieldMapping) => {
        const hasValidSource = sourceFields.includes(record.source_field);
        const hasValidTarget = targetFields.some(f => f.value === record.target_field);
        
        if (hasValidSource && hasValidTarget) {
          return <Tag color="success" icon={<CheckCircleOutlined />}>Valid</Tag>;
        } else {
          return <Tag color="error" icon={<ExclamationCircleOutlined />}>Invalid</Tag>;
        }
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: FieldMapping, index: number) => (
        <Space>
          <Tooltip title="Test Transformation">
            <Button
              type="text"
              size="small"
              icon={<ExperimentOutlined />}
              onClick={() => testTransformation(record)}
            />
          </Tooltip>
          <Tooltip title="Edit Mapping">
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={() => editMapping(record, index)}
            />
          </Tooltip>
          <Tooltip title="Duplicate Mapping">
            <Button
              type="text"
              size="small"
              icon={<CopyOutlined />}
              onClick={() => duplicateMapping(record)}
            />
          </Tooltip>
          <Tooltip title="Delete Mapping">
            <Button
              type="text"
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => deleteMapping(index)}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  // Initialize form when editing
  useEffect(() => {
    if (editingMapping && editModalVisible) {
      form.setFieldsValue(editingMapping);
    }
  }, [editingMapping, editModalVisible, form]);

  return (
    <div className={`api-mapping-config ${className || ''}`}>
      <Card 
        title="Field Mappings Configuration"
        extra={
          <Button type="primary" icon={<PlusOutlined />} onClick={addMapping}>
            Add Mapping
          </Button>
        }
      >
        <Alert
          message="Field Mapping Information"
          description="Configure how your local data fields map to external API fields. Transformations allow you to modify values before sending."
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Table
          columns={columns}
          dataSource={mappings.map((mapping, index) => ({ ...mapping, key: index }))}
          size="small"
          pagination={false}
          locale={{
            emptyText: 'No field mappings configured. Click "Add Mapping" to create one.'
          }}
        />
      </Card>

      {/* Edit/Create Mapping Modal */}
      <Modal
        title={editingIndex >= 0 ? 'Edit Field Mapping' : 'Create Field Mapping'}
        open={editModalVisible}
        onCancel={() => setEditModalVisible(false)}
        onOk={saveMapping}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Row gutter={16}>
            <Col xs={24} md={12}>
              <Form.Item
                name="source_field"
                label="Source Field"
                rules={[{ required: true, message: 'Please select source field' }]}
              >
                <Select
                  placeholder="Select source field"
                  showSearch
                  filterOption={(input, option) =>
                    (option?.label as string)?.toLowerCase().includes(input.toLowerCase()) || false
                  }
                >
                  {sourceFields.map(field => (
                    <Option key={field} value={field}>
                      <div>
                        {field}
                        <Tag color="blue" style={{ marginLeft: 8 }}>
                          {getFieldType(field)}
                        </Tag>
                      </div>
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item
                name="target_field"
                label="Target Field"
                rules={[{ required: true, message: 'Please select target field' }]}
              >
                <Select
                  placeholder="Select target field"
                  showSearch
                  filterOption={(input, option) =>
                    (option?.label as string)?.toLowerCase().includes(input.toLowerCase()) || false
                  }
                >
                  {targetFields.map(field => (
                    <Option key={field.value} value={field.value}>
                      <div>
                        {field.label}
                        <br />
                        <Text type="secondary" style={{ fontSize: 11 }}>{field.value}</Text>
                      </div>
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="transformation"
            label={
              <div>
                Transformation (Optional)
                <Tooltip title="JavaScript expression to transform the value. Use 'value' to reference the input.">
                  <InfoCircleOutlined style={{ marginLeft: 4 }} />
                </Tooltip>
              </div>
            }
          >
            <Input.TextArea 
              rows={2}
              placeholder="e.g., value * 100, Math.round(value), new Date(value).toISOString()"
            />
          </Form.Item>

          <Form.Item label="Common Transformations">
            <Select
              placeholder="Choose a template..."
              onChange={(value) => form.setFieldsValue({ transformation: value })}
              allowClear
            >
              {transformationTemplates.map(template => (
                <Option key={template.value} value={template.value}>
                  {template.label}
                </Option>
              ))}
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* Transformation Test Modal */}
      <Modal
        title="Test Transformation"
        open={testModalVisible}
        onCancel={() => setTestModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setTestModalVisible(false)}>
            Close
          </Button>
        ]}
        width={700}
      >
        {editingMapping && (
          <div>
            <Alert
              message="Transformation Test"
              description={`Testing transformation: ${editingMapping.transformation || 'No transformation'}`}
              type="info"
              style={{ marginBottom: 16 }}
            />
            
            <Table
              dataSource={testResults.map((result, index) => ({ ...result, key: index }))}
              size="small"
              pagination={false}
              columns={[
                {
                  title: 'Input',
                  dataIndex: 'input',
                  key: 'input',
                  render: (value: any) => (
                    <Text code>{JSON.stringify(value)}</Text>
                  )
                },
                {
                  title: 'Output',
                  dataIndex: 'output',
                  key: 'output',
                  render: (value: any, record: TransformationTest) => (
                    record.success ? (
                      <Text code style={{ color: '#52c41a' }}>{JSON.stringify(value)}</Text>
                    ) : (
                      <Text type="danger">Error</Text>
                    )
                  )
                },
                {
                  title: 'Status',
                  key: 'status',
                  render: (record: TransformationTest) => (
                    record.success ? (
                      <Tag color="success" icon={<CheckCircleOutlined />}>Success</Tag>
                    ) : (
                      <Tooltip title={record.error}>
                        <Tag color="error" icon={<ExclamationCircleOutlined />}>Failed</Tag>
                      </Tooltip>
                    )
                  )
                }
              ]}
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default ApiMappingConfig;