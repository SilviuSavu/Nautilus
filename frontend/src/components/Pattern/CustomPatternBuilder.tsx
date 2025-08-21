/**
 * Custom Pattern Builder Component
 * Allows users to create custom chart pattern definitions
 */

import React, { useState, useCallback } from 'react'
import { 
  Card, 
  Form, 
  Input, 
  Select, 
  Button, 
  Space, 
  InputNumber, 
  Divider, 
  List, 
  Modal, 
  Tag,
  Switch,
  Slider,
  message,
  Collapse,
  Tooltip
} from 'antd'
import { 
  PlusOutlined, 
  DeleteOutlined, 
  SaveOutlined, 
  TestOutlined,
  InfoCircleOutlined,
  SettingOutlined
} from '@ant-design/icons'
import { PatternDefinition, PatternRule } from '../../types/charting'
import { patternRecognition } from '../../services/patternRecognition'

const { Option } = Select
const { TextArea } = Input
const { Panel } = Collapse

interface CustomPatternBuilderProps {
  onPatternCreated?: (patternId: string) => void
  className?: string
}

interface RuleBuilder {
  id: string
  type: PatternRule['type']
  condition: string
  parameters: Record<string, any>
  weight: number
}

export const CustomPatternBuilder: React.FC<CustomPatternBuilderProps> = ({
  onPatternCreated,
  className
}) => {
  const [form] = Form.useForm()
  const [rules, setRules] = useState<RuleBuilder[]>([])
  const [isTestModalVisible, setIsTestModalVisible] = useState(false)
  const [testResults, setTestResults] = useState<any>(null)
  const [isAdvancedMode, setIsAdvancedMode] = useState(false)

  const addRule = useCallback(() => {
    const newRule: RuleBuilder = {
      id: `rule_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: 'price_action',
      condition: '',
      parameters: {},
      weight: 1
    }
    setRules(prev => [...prev, newRule])
  }, [])

  const updateRule = useCallback((ruleId: string, updates: Partial<RuleBuilder>) => {
    setRules(prev => prev.map(rule => 
      rule.id === ruleId ? { ...rule, ...updates } : rule
    ))
  }, [])

  const removeRule = useCallback((ruleId: string) => {
    setRules(prev => prev.filter(rule => rule.id !== ruleId))
  }, [])

  const handleSavePattern = async () => {
    try {
      const values = await form.validateFields()
      
      if (rules.length === 0) {
        message.error('Please add at least one rule to the pattern')
        return
      }

      const patternDefinition: Omit<PatternDefinition, 'id'> = {
        name: values.name,
        type: values.type,
        rules: rules.map(rule => ({
          type: rule.type,
          condition: rule.condition,
          parameters: rule.parameters,
          weight: rule.weight
        })),
        minBars: values.minBars || 10,
        maxBars: values.maxBars || 100,
        minConfidence: values.minConfidence || 0.6,
        description: values.description,
        category: values.category || 'custom',
        tags: values.tags ? values.tags.split(',').map((tag: string) => tag.trim()) : []
      }

      const patternId = patternRecognition.registerPattern(patternDefinition)
      
      message.success(`Pattern "${values.name}" created successfully!`)
      form.resetFields()
      setRules([])
      onPatternCreated?.(patternId)
      
    } catch (error) {
      message.error('Failed to create pattern')
      console.error('Pattern creation error:', error)
    }
  }

  const handleTestPattern = async () => {
    try {
      const values = await form.validateFields()
      
      if (rules.length === 0) {
        message.error('Please add at least one rule to test')
        return
      }

      // Create temporary pattern for testing
      const testPattern: Omit<PatternDefinition, 'id'> = {
        name: values.name || 'Test Pattern',
        type: values.type || 'custom',
        rules: rules.map(rule => ({
          type: rule.type,
          condition: rule.condition,
          parameters: rule.parameters,
          weight: rule.weight
        })),
        minBars: values.minBars || 10,
        maxBars: values.maxBars || 100,
        minConfidence: values.minConfidence || 0.6
      }

      // TODO: Implement pattern testing with sample data
      setTestResults({
        confidence: Math.random() * 0.4 + 0.6, // Mock confidence
        matchedRules: rules.length,
        totalRules: rules.length,
        suggestions: [
          'Consider adjusting the minimum confidence threshold',
          'Volume rules could improve accuracy'
        ]
      })
      
      setIsTestModalVisible(true)
      
    } catch (error) {
      message.error('Pattern validation failed')
    }
  }

  const getRuleConditionOptions = (ruleType: PatternRule['type']) => {
    switch (ruleType) {
      case 'price_action':
        return [
          'peak_sequence', 'double_peak', 'double_valley', 'ascending_triangle',
          'descending_triangle', 'cup_formation', 'handle_formation', 'strong_move_up',
          'strong_move_down', 'flag_consolidation', 'converging_lines', 'support_break',
          'resistance_break', 'trend_continuation', 'trend_reversal'
        ]
      case 'volume':
        return [
          'decreasing', 'increasing', 'spike', 'lower_on_second', 'higher_on_second',
          'above_average', 'below_average', 'volume_confirmation'
        ]
      case 'indicator':
        return [
          'rsi_divergence', 'macd_divergence', 'stochastic_crossover',
          'bollinger_squeeze', 'moving_average_cross', 'momentum_shift'
        ]
      default:
        return []
    }
  }

  const getParameterFields = (ruleType: PatternRule['type'], condition: string) => {
    const commonParams = {
      price_action: {
        peak_sequence: ['peaks', 'order', 'symmetry_threshold'],
        double_peak: ['tolerance', 'min_separation'],
        double_valley: ['tolerance', 'min_separation'],
        ascending_triangle: ['minTouches', 'slope_tolerance'],
        cup_formation: ['depth', 'symmetry', 'duration'],
        handle_formation: ['retracement', 'duration'],
        strong_move_up: ['minMove', 'timeframe'],
        flag_consolidation: ['slope', 'duration', 'volume_pattern'],
        converging_lines: ['direction', 'convergence_angle']
      },
      volume: {
        decreasing: ['lookback_period', 'min_decrease'],
        increasing: ['lookback_period', 'min_increase'],
        spike: ['threshold_multiplier', 'duration'],
        lower_on_second: ['comparison_period'],
        higher_on_second: ['comparison_period']
      },
      indicator: {
        rsi_divergence: ['rsi_period', 'divergence_type'],
        macd_divergence: ['fast_period', 'slow_period', 'signal_period'],
        moving_average_cross: ['fast_period', 'slow_period']
      }
    }

    return commonParams[ruleType]?.[condition] || []
  }

  return (
    <Card 
      className={className} 
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Custom Pattern Builder</span>
          <Space>
            <Switch 
              size="small"
              checked={isAdvancedMode}
              onChange={setIsAdvancedMode}
              checkedChildren="Advanced"
              unCheckedChildren="Simple"
            />
          </Space>
        </div>
      } 
      size="small"
    >
      <Form form={form} layout="vertical" size="small">
        <div style={{ display: 'grid', gap: 12, gridTemplateColumns: isAdvancedMode ? '1fr 1fr' : '1fr' }}>
          {/* Basic Pattern Information */}
          <div>
            <Form.Item
              name="name"
              label="Pattern Name"
              rules={[{ required: true, message: 'Please enter pattern name' }]}
            >
              <Input placeholder="e.g., Custom Bull Flag" />
            </Form.Item>

            <Form.Item name="description" label="Description">
              <TextArea 
                rows={2} 
                placeholder="Describe what this pattern identifies..."
              />
            </Form.Item>

            <div style={{ display: 'grid', gap: 8, gridTemplateColumns: '1fr 1fr' }}>
              <Form.Item
                name="type"
                label="Pattern Type"
                rules={[{ required: true }]}
              >
                <Select placeholder="Select type">
                  <Option value="custom">Custom</Option>
                  <Option value="reversal">Reversal</Option>
                  <Option value="continuation">Continuation</Option>
                  <Option value="consolidation">Consolidation</Option>
                </Select>
              </Form.Item>

              <Form.Item name="category" label="Category">
                <Select placeholder="Category">
                  <Option value="custom">Custom</Option>
                  <Option value="trading">Trading</Option>
                  <Option value="analysis">Analysis</Option>
                  <Option value="experimental">Experimental</Option>
                </Select>
              </Form.Item>
            </div>

            <Form.Item name="tags" label="Tags">
              <Input placeholder="trend, reversal, bullish (comma-separated)" />
            </Form.Item>
          </div>

          {/* Advanced Configuration */}
          {isAdvancedMode && (
            <div>
              <div style={{ display: 'grid', gap: 8, gridTemplateColumns: '1fr 1fr' }}>
                <Form.Item name="minBars" label="Min Bars">
                  <InputNumber min={5} max={50} defaultValue={10} style={{ width: '100%' }} />
                </Form.Item>

                <Form.Item name="maxBars" label="Max Bars">
                  <InputNumber min={20} max={500} defaultValue={100} style={{ width: '100%' }} />
                </Form.Item>
              </div>

              <Form.Item name="minConfidence" label="Min Confidence">
                <Slider
                  min={0.1}
                  max={1}
                  step={0.05}
                  defaultValue={0.6}
                  marks={{ 0.1: '10%', 0.5: '50%', 1: '100%' }}
                />
              </Form.Item>
            </div>
          )}
        </div>

        <Divider>Pattern Rules</Divider>

        {/* Rules Builder */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <span style={{ fontSize: '14px', fontWeight: 500 }}>
              Rules ({rules.length})
            </span>
            <Button 
              type="dashed" 
              size="small" 
              icon={<PlusOutlined />}
              onClick={addRule}
            >
              Add Rule
            </Button>
          </div>

          {rules.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              padding: 20, 
              border: '1px dashed #d9d9d9', 
              borderRadius: 6,
              color: '#999'
            }}>
              No rules defined. Add your first rule to get started.
            </div>
          ) : (
            <List
              size="small"
              dataSource={rules}
              renderItem={(rule) => (
                <List.Item
                  style={{ 
                    border: '1px solid #f0f0f0', 
                    borderRadius: 6, 
                    marginBottom: 8,
                    padding: 12
                  }}
                >
                  <div style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <Tag color="blue">Rule {rules.findIndex(r => r.id === rule.id) + 1}</Tag>
                      <Button
                        type="link"
                        size="small"
                        danger
                        icon={<DeleteOutlined />}
                        onClick={() => removeRule(rule.id)}
                      />
                    </div>

                    <div style={{ display: 'grid', gap: 8, gridTemplateColumns: isAdvancedMode ? '1fr 1fr 1fr' : '1fr 1fr' }}>
                      <div>
                        <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
                          Rule Type:
                        </label>
                        <Select
                          size="small"
                          value={rule.type}
                          onChange={(value) => updateRule(rule.id, { type: value, condition: '', parameters: {} })}
                          style={{ width: '100%' }}
                        >
                          <Option value="price_action">Price Action</Option>
                          <Option value="volume">Volume</Option>
                          <Option value="indicator">Indicator</Option>
                        </Select>
                      </div>

                      <div>
                        <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
                          Condition:
                        </label>
                        <Select
                          size="small"
                          value={rule.condition}
                          onChange={(value) => updateRule(rule.id, { condition: value, parameters: {} })}
                          style={{ width: '100%' }}
                          placeholder="Select condition"
                        >
                          {getRuleConditionOptions(rule.type).map(condition => (
                            <Option key={condition} value={condition}>
                              {condition.replace(/_/g, ' ')}
                            </Option>
                          ))}
                        </Select>
                      </div>

                      {isAdvancedMode && (
                        <div>
                          <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
                            Weight:
                          </label>
                          <InputNumber
                            size="small"
                            min={0.1}
                            max={2}
                            step={0.1}
                            value={rule.weight}
                            onChange={(value) => updateRule(rule.id, { weight: value || 1 })}
                            style={{ width: '100%' }}
                          />
                        </div>
                      )}
                    </div>

                    {/* Dynamic parameter fields */}
                    {rule.condition && getParameterFields(rule.type, rule.condition).length > 0 && (
                      <div style={{ marginTop: 8 }}>
                        <Collapse size="small">
                          <Panel 
                            header={`Parameters (${getParameterFields(rule.type, rule.condition).length})`}
                            key="params"
                          >
                            <div style={{ display: 'grid', gap: 8, gridTemplateColumns: '1fr 1fr' }}>
                              {getParameterFields(rule.type, rule.condition).map(param => (
                                <div key={param}>
                                  <label style={{ fontSize: '10px', display: 'block', marginBottom: 2 }}>
                                    {param.replace(/_/g, ' ')}:
                                  </label>
                                  <InputNumber
                                    size="small"
                                    value={rule.parameters[param]}
                                    onChange={(value) => 
                                      updateRule(rule.id, { 
                                        parameters: { ...rule.parameters, [param]: value }
                                      })
                                    }
                                    style={{ width: '100%' }}
                                    placeholder="Value"
                                  />
                                </div>
                              ))}
                            </div>
                          </Panel>
                        </Collapse>
                      </div>
                    )}
                  </div>
                </List.Item>
              )}
            />
          )}
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
          <Space>
            <Button 
              icon={<TestOutlined />} 
              onClick={handleTestPattern}
              disabled={rules.length === 0}
            >
              Test Pattern
            </Button>
            
            <Tooltip title="Advanced configuration options">
              <Button 
                icon={<SettingOutlined />} 
                onClick={() => setIsAdvancedMode(!isAdvancedMode)}
              />
            </Tooltip>
          </Space>

          <Button 
            type="primary" 
            icon={<SaveOutlined />}
            onClick={handleSavePattern}
            disabled={rules.length === 0}
          >
            Save Pattern
          </Button>
        </div>
      </Form>

      {/* Test Results Modal */}
      <Modal
        title="Pattern Test Results"
        open={isTestModalVisible}
        onCancel={() => setIsTestModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setIsTestModalVisible(false)}>
            Close
          </Button>
        ]}
      >
        {testResults && (
          <div style={{ display: 'grid', gap: 12 }}>
            <div>
              <strong>Confidence Score:</strong>
              <div style={{ 
                fontSize: '24px', 
                color: testResults.confidence > 0.7 ? '#52c41a' : '#faad14',
                fontWeight: 'bold'
              }}>
                {(testResults.confidence * 100).toFixed(1)}%
              </div>
            </div>

            <div>
              <strong>Rules Matched:</strong> {testResults.matchedRules} / {testResults.totalRules}
            </div>

            <div>
              <strong>Suggestions:</strong>
              <ul>
                {testResults.suggestions.map((suggestion: string, index: number) => (
                  <li key={index}>{suggestion}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </Modal>
    </Card>
  )
}

export default CustomPatternBuilder