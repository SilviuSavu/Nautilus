/**
 * Parameter Configuration Component
 * Provides detailed configuration interface for indicator parameters
 */

import React, { useState } from 'react'
import { Card, Form, InputNumber, Select, Switch, ColorPicker, Slider, Button, Space, Tooltip, Collapse } from 'antd'
import { InfoCircleOutlined, ReloadOutlined, SaveOutlined } from '@ant-design/icons'
import { IndicatorParameter, TechnicalIndicator } from '../../services/indicatorEngine'

const { Option } = Select
const { Panel } = Collapse

interface ParameterConfigProps {
  indicator: TechnicalIndicator
  currentParameters: Record<string, any>
  onChange: (parameters: Record<string, any>) => void
  onSavePreset?: (name: string, parameters: Record<string, any>) => void
  onLoadPreset?: (parameters: Record<string, any>) => void
  presets?: Array<{ name: string; parameters: Record<string, any> }>
}

export const ParameterConfig: React.FC<ParameterConfigProps> = ({
  indicator,
  currentParameters,
  onChange,
  onSavePreset,
  onLoadPreset,
  presets = []
}) => {
  const [form] = Form.useForm()
  const [presetName, setPresetName] = useState('')

  const handleParameterChange = (paramName: string, value: any) => {
    const newParameters = {
      ...currentParameters,
      [paramName]: value
    }
    onChange(newParameters)
  }

  const handleResetToDefaults = () => {
    const defaultParams: Record<string, any> = {}
    indicator.parameters.forEach(param => {
      defaultParams[param.name] = param.defaultValue
    })
    onChange(defaultParams)
    form.setFieldsValue(defaultParams)
  }

  const handleSavePreset = () => {
    if (presetName && onSavePreset) {
      onSavePreset(presetName, currentParameters)
      setPresetName('')
    }
  }

  const handleLoadPreset = (preset: { name: string; parameters: Record<string, any> }) => {
    onChange(preset.parameters)
    form.setFieldsValue(preset.parameters)
    if (onLoadPreset) {
      onLoadPreset(preset.parameters)
    }
  }

  const renderParameterControl = (param: IndicatorParameter) => {
    const value = currentParameters[param.name] ?? param.defaultValue

    switch (param.type) {
      case 'number':
        // Use slider for bounded numeric values, InputNumber for unbounded
        if (param.min !== undefined && param.max !== undefined) {
          return (
            <div>
              <Slider
                min={param.min}
                max={param.max}
                value={value}
                onChange={(val) => handleParameterChange(param.name, val)}
                style={{ marginBottom: 8 }}
              />
              <InputNumber
                min={param.min}
                max={param.max}
                value={value}
                onChange={(val) => handleParameterChange(param.name, val)}
                size="small"
                style={{ width: '100%' }}
              />
            </div>
          )
        }
        return (
          <InputNumber
            min={param.min}
            max={param.max}
            value={value}
            onChange={(val) => handleParameterChange(param.name, val)}
            style={{ width: '100%' }}
          />
        )

      case 'boolean':
        return (
          <Switch
            checked={value}
            onChange={(val) => handleParameterChange(param.name, val)}
            checkedChildren="ON"
            unCheckedChildren="OFF"
          />
        )

      case 'color':
        return (
          <ColorPicker
            value={value}
            onChange={(color) => handleParameterChange(param.name, color.toHexString())}
            showText
          />
        )

      case 'string':
        if (param.options) {
          return (
            <Select
              value={value}
              onChange={(val) => handleParameterChange(param.name, val)}
              style={{ width: '100%' }}
            >
              {param.options.map(option => (
                <Option key={option} value={option}>{option}</Option>
              ))}
            </Select>
          )
        }
        return (
          <input
            type="text"
            value={value}
            onChange={(e) => handleParameterChange(param.name, e.target.value)}
            className="ant-input"
            style={{ width: '100%' }}
          />
        )

      default:
        return <span>Unsupported parameter type: {param.type}</span>
    }
  }

  const getParameterDescription = (param: IndicatorParameter): string => {
    const descriptions: Record<string, string> = {
      period: 'Number of periods to use in the calculation',
      fastPeriod: 'Fast moving average period',
      slowPeriod: 'Slow moving average period',
      signalPeriod: 'Signal line smoothing period',
      stdDev: 'Number of standard deviations for bands',
      source: 'Price source for the calculation (close, open, high, low, volume)',
      smoothing: 'Smoothing factor for the calculation',
      threshold: 'Threshold level for signals',
      sensitivity: 'Sensitivity level for pattern detection'
    }
    return descriptions[param.name] || `Configuration parameter: ${param.name}`
  }

  return (
    <Card title={`${indicator.name} Configuration`} size="small">
      <Form
        form={form}
        layout="vertical"
        initialValues={currentParameters}
        size="small"
      >
        {/* Parameter Controls */}
        <div style={{ marginBottom: 16 }}>
          {indicator.parameters.map(param => (
            <Form.Item
              key={param.name}
              label={
                <Space>
                  <span style={{ textTransform: 'capitalize' }}>{param.name}</span>
                  <Tooltip title={getParameterDescription(param)}>
                    <InfoCircleOutlined style={{ color: '#999' }} />
                  </Tooltip>
                </Space>
              }
              style={{ marginBottom: 12 }}
            >
              {renderParameterControl(param)}
            </Form.Item>
          ))}
        </div>

        {/* Action Buttons */}
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={handleResetToDefaults}
          >
            Reset to Defaults
          </Button>
        </Space>

        {/* Presets Section */}
        {(presets.length > 0 || onSavePreset) && (
          <Collapse size="small" style={{ marginTop: 16 }}>
            <Panel header="Presets" key="presets">
              {/* Load Presets */}
              {presets.length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <label style={{ display: 'block', marginBottom: 8, fontSize: '12px', fontWeight: 500 }}>
                    Load Preset:
                  </label>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {presets.map(preset => (
                      <Button
                        key={preset.name}
                        size="small"
                        type="text"
                        onClick={() => handleLoadPreset(preset)}
                        style={{ 
                          textAlign: 'left', 
                          height: 'auto',
                          padding: '4px 8px'
                        }}
                      >
                        {preset.name}
                      </Button>
                    ))}
                  </div>
                </div>
              )}

              {/* Save Preset */}
              {onSavePreset && (
                <div>
                  <label style={{ display: 'block', marginBottom: 8, fontSize: '12px', fontWeight: 500 }}>
                    Save Current Settings:
                  </label>
                  <Space.Compact style={{ width: '100%' }}>
                    <input
                      type="text"
                      value={presetName}
                      onChange={(e) => setPresetName(e.target.value)}
                      placeholder="Preset name"
                      className="ant-input"
                      style={{ flex: 1 }}
                    />
                    <Button
                      size="small"
                      type="primary"
                      icon={<SaveOutlined />}
                      onClick={handleSavePreset}
                      disabled={!presetName}
                    >
                      Save
                    </Button>
                  </Space.Compact>
                </div>
              )}
            </Panel>
          </Collapse>
        )}
      </Form>

      {/* Quick Info */}
      <div style={{ 
        marginTop: 16, 
        padding: 8, 
        background: '#f5f5f5', 
        borderRadius: 4,
        fontSize: '11px',
        color: '#666'
      }}>
        <div><strong>Type:</strong> {indicator.type === 'built_in' ? 'Built-in' : 'Custom'}</div>
        <div><strong>Overlay:</strong> {indicator.display.overlay ? 'Yes' : 'No'}</div>
        {indicator.calculation.source && (
          <div><strong>Source:</strong> {indicator.calculation.source}</div>
        )}
      </div>
    </Card>
  )
}

export default ParameterConfig