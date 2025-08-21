/**
 * Advanced Indicator Builder Component
 * Provides interface for creating and managing technical indicators
 */

import React, { useState, useEffect } from 'react'
import { Card, Select, InputNumber, ColorPicker, Switch, Button, Space, Divider, Modal, message } from 'antd'
import { PlusOutlined, DeleteOutlined, EyeOutlined, EyeInvisibleOutlined, SettingOutlined } from '@ant-design/icons'
import { indicatorEngine, TechnicalIndicator, IndicatorParameter } from '../../services/indicatorEngine'
import { useChartStore } from '../Chart/hooks/useChartStore'

const { Option } = Select

interface IndicatorBuilderProps {
  className?: string
  onIndicatorAdd?: (indicatorId: string, params: Record<string, any>) => void
  onIndicatorRemove?: (instanceId: string) => void
}

interface ActiveIndicatorInstance {
  id: string
  indicatorId: string
  name: string
  parameters: Record<string, any>
  visible: boolean
  color: string
}

export const IndicatorBuilder: React.FC<IndicatorBuilderProps> = ({
  className,
  onIndicatorAdd,
  onIndicatorRemove
}) => {
  const [availableIndicators, setAvailableIndicators] = useState<TechnicalIndicator[]>([])
  const [activeIndicators, setActiveIndicators] = useState<ActiveIndicatorInstance[]>([])
  const [selectedIndicatorId, setSelectedIndicatorId] = useState<string>('')
  const [parameters, setParameters] = useState<Record<string, any>>({})
  const [isCustomIndicatorModalVisible, setIsCustomIndicatorModalVisible] = useState(false)
  const [customIndicatorScript, setCustomIndicatorScript] = useState('')
  const [customIndicatorName, setCustomIndicatorName] = useState('')

  useEffect(() => {
    // Load available indicators from engine
    const indicators = indicatorEngine.getAvailableIndicators()
    setAvailableIndicators(indicators)
    
    if (indicators.length > 0 && !selectedIndicatorId) {
      setSelectedIndicatorId(indicators[0].id)
      initializeParameters(indicators[0])
    }
  }, [selectedIndicatorId])

  const initializeParameters = (indicator: TechnicalIndicator) => {
    const defaultParams: Record<string, any> = {}
    indicator.parameters.forEach(param => {
      defaultParams[param.name] = param.defaultValue
    })
    setParameters(defaultParams)
  }

  const handleIndicatorSelect = (indicatorId: string) => {
    setSelectedIndicatorId(indicatorId)
    const indicator = availableIndicators.find(ind => ind.id === indicatorId)
    if (indicator) {
      initializeParameters(indicator)
    }
  }

  const handleParameterChange = (paramName: string, value: any) => {
    setParameters(prev => ({
      ...prev,
      [paramName]: value
    }))
  }

  const handleAddIndicator = () => {
    if (!selectedIndicatorId) return

    const indicator = availableIndicators.find(ind => ind.id === selectedIndicatorId)
    if (!indicator) return

    const instanceId = `${selectedIndicatorId}_${Date.now()}`
    const newInstance: ActiveIndicatorInstance = {
      id: instanceId,
      indicatorId: selectedIndicatorId,
      name: `${indicator.name}(${parameters.period || 'custom'})`,
      parameters: { ...parameters },
      visible: true,
      color: indicator.display.color
    }

    setActiveIndicators(prev => [...prev, newInstance])
    onIndicatorAdd?.(selectedIndicatorId, parameters)
    message.success(`Added ${indicator.name}`)
  }

  const handleRemoveIndicator = (instanceId: string) => {
    setActiveIndicators(prev => prev.filter(ind => ind.id !== instanceId))
    onIndicatorRemove?.(instanceId)
    message.success('Indicator removed')
  }

  const handleToggleVisibility = (instanceId: string) => {
    setActiveIndicators(prev =>
      prev.map(ind =>
        ind.id === instanceId ? { ...ind, visible: !ind.visible } : ind
      )
    )
  }

  const handleColorChange = (instanceId: string, color: string) => {
    setActiveIndicators(prev =>
      prev.map(ind =>
        ind.id === instanceId ? { ...ind, color } : ind
      )
    )
  }

  const handleCreateCustomIndicator = () => {
    if (!customIndicatorName || !customIndicatorScript) {
      message.error('Please provide both name and script for custom indicator')
      return
    }

    try {
      const indicatorId = indicatorEngine.createCustomIndicator(
        customIndicatorName,
        customIndicatorScript,
        [
          { name: 'period', type: 'number', defaultValue: 20, min: 1, max: 200 }
        ],
        {
          color: '#FF6B6B',
          lineWidth: 2,
          style: 'solid',
          overlay: true
        }
      )

      // Refresh available indicators
      const updatedIndicators = indicatorEngine.getAvailableIndicators()
      setAvailableIndicators(updatedIndicators)
      setSelectedIndicatorId(indicatorId)
      
      setIsCustomIndicatorModalVisible(false)
      setCustomIndicatorName('')
      setCustomIndicatorScript('')
      message.success('Custom indicator created successfully')
    } catch (error) {
      message.error(`Error creating custom indicator: ${error.message}`)
    }
  }

  const renderParameterInput = (param: IndicatorParameter, value: any) => {
    switch (param.type) {
      case 'number':
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
          />
        )
      case 'color':
        return (
          <ColorPicker
            value={value}
            onChange={(color) => handleParameterChange(param.name, color.toHexString())}
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
            style={{ width: '100%', padding: '4px 8px', border: '1px solid #d9d9d9', borderRadius: '4px' }}
          />
        )
      default:
        return null
    }
  }

  const selectedIndicator = availableIndicators.find(ind => ind.id === selectedIndicatorId)

  return (
    <div className={className}>
      <Card 
        title="Technical Indicators" 
        size="small"
        extra={
          <Button
            size="small"
            type="link"
            icon={<PlusOutlined />}
            onClick={() => setIsCustomIndicatorModalVisible(true)}
          >
            Custom
          </Button>
        }
      >
        {/* Indicator Selection */}
        <div style={{ marginBottom: 16 }}>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>
            Select Indicator:
          </label>
          <Select
            value={selectedIndicatorId}
            onChange={handleIndicatorSelect}
            style={{ width: '100%' }}
            placeholder="Choose an indicator"
          >
            {availableIndicators.map(indicator => (
              <Option key={indicator.id} value={indicator.id}>
                <Space>
                  <span>{indicator.name}</span>
                  {indicator.type === 'custom' && <span style={{ color: '#1890ff' }}>(Custom)</span>}
                  {indicator.type === 'scripted' && <span style={{ color: '#52c41a' }}>(Script)</span>}
                </Space>
              </Option>
            ))}
          </Select>
        </div>

        {/* Parameter Configuration */}
        {selectedIndicator && (
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>
              Parameters:
            </label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {selectedIndicator.parameters.map(param => (
                <div key={param.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <label style={{ minWidth: 80, fontSize: '12px' }}>
                    {param.name}:
                  </label>
                  <div style={{ flex: 1 }}>
                    {renderParameterInput(param, parameters[param.name])}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Add Button */}
        <Button
          type="primary"
          block
          icon={<PlusOutlined />}
          onClick={handleAddIndicator}
          disabled={!selectedIndicatorId}
        >
          Add Indicator
        </Button>

        <Divider />

        {/* Active Indicators */}
        <div>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>
            Active Indicators ({activeIndicators.length}):
          </label>
          
          {activeIndicators.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              padding: 20, 
              color: '#999',
              background: '#fafafa',
              borderRadius: 4,
              border: '1px dashed #d9d9d9'
            }}>
              No indicators added yet
            </div>
          ) : (
            <div style={{ maxHeight: 200, overflowY: 'auto' }}>
              {activeIndicators.map(instance => (
                <div
                  key={instance.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    marginBottom: 8,
                    background: '#f5f5f5',
                    borderRadius: 4,
                    border: '1px solid #e8e8e8'
                  }}
                >
                  <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div
                      style={{
                        width: 12,
                        height: 12,
                        backgroundColor: instance.color,
                        borderRadius: '50%',
                        border: '1px solid #ccc'
                      }}
                    />
                    <span style={{ fontSize: '12px', fontWeight: 500 }}>
                      {instance.name}
                    </span>
                  </div>
                  
                  <Space size="small">
                    <ColorPicker
                      value={instance.color}
                      onChange={(color) => handleColorChange(instance.id, color.toHexString())}
                      size="small"
                    />
                    <Button
                      size="small"
                      type="text"
                      icon={instance.visible ? <EyeOutlined /> : <EyeInvisibleOutlined />}
                      onClick={() => handleToggleVisibility(instance.id)}
                    />
                    <Button
                      size="small"
                      type="text"
                      danger
                      icon={<DeleteOutlined />}
                      onClick={() => handleRemoveIndicator(instance.id)}
                    />
                  </Space>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* Custom Indicator Modal */}
      <Modal
        title="Create Custom Indicator"
        open={isCustomIndicatorModalVisible}
        onOk={handleCreateCustomIndicator}
        onCancel={() => setIsCustomIndicatorModalVisible(false)}
        width={600}
      >
        <div style={{ marginBottom: 16 }}>
          <label style={{ display: 'block', marginBottom: 8 }}>Name:</label>
          <input
            type="text"
            value={customIndicatorName}
            onChange={(e) => setCustomIndicatorName(e.target.value)}
            placeholder="Enter indicator name"
            style={{ 
              width: '100%', 
              padding: '8px 12px', 
              border: '1px solid #d9d9d9', 
              borderRadius: '4px' 
            }}
          />
        </div>
        <div>
          <label style={{ display: 'block', marginBottom: 8 }}>JavaScript Code:</label>
          <textarea
            value={customIndicatorScript}
            onChange={(e) => setCustomIndicatorScript(e.target.value)}
            placeholder={`// Example: Simple custom indicator
// Available variables: data (OHLCV array), params (parameters)
// Available functions: sma(), ema(), rsi()

const closes = data.map(d => d.close);
return sma(closes, params.period || 20);`}
            style={{ 
              width: '100%', 
              height: 200, 
              padding: '8px 12px', 
              border: '1px solid #d9d9d9', 
              borderRadius: '4px',
              fontFamily: 'Monaco, monospace',
              fontSize: '12px'
            }}
          />
        </div>
      </Modal>
    </div>
  )
}

export default IndicatorBuilder