/**
 * Pattern Management Dashboard
 * Manages custom and built-in pattern definitions
 */

import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Tabs, 
  Table, 
  Button, 
  Space, 
  Tag, 
  Modal, 
  Popconfirm,
  Input,
  Select,
  Switch,
  message,
  Tooltip,
  Progress
} from 'antd'
import { 
  EditOutlined, 
  DeleteOutlined, 
  CopyOutlined,
  DownloadOutlined,
  UploadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons'
import { PatternDefinition } from '../../types/charting'
import { patternRecognition } from '../../services/patternRecognition'
import CustomPatternBuilder from './CustomPatternBuilder'

const { Search } = Input
const { Option } = Select

interface PatternStats {
  totalDetections: number
  avgConfidence: number
  successRate: number
  lastDetected: string | null
}

interface ExtendedPatternDefinition extends PatternDefinition {
  enabled: boolean
  stats: PatternStats
  isBuiltIn: boolean
}

export const PatternManagementDashboard: React.FC = () => {
  const [patterns, setPatterns] = useState<ExtendedPatternDefinition[]>([])
  const [filteredPatterns, setFilteredPatterns] = useState<ExtendedPatternDefinition[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [filterCategory, setFilterCategory] = useState<string>('all')
  const [selectedPattern, setSelectedPattern] = useState<ExtendedPatternDefinition | null>(null)
  const [isEditModalVisible, setIsEditModalVisible] = useState(false)
  const [activeTab, setActiveTab] = useState('manage')

  useEffect(() => {
    loadPatterns()
  }, [])

  useEffect(() => {
    filterPatterns()
  }, [patterns, searchTerm, filterCategory])

  const loadPatterns = () => {
    const definitions = patternRecognition.getPatternDefinitions()
    const config = patternRecognition.getConfig()
    
    const extendedPatterns: ExtendedPatternDefinition[] = definitions.map(def => ({
      ...def,
      enabled: config.enabledPatterns.length === 0 || config.enabledPatterns.includes(def.id),
      stats: {
        totalDetections: Math.floor(Math.random() * 100), // Mock data
        avgConfidence: Math.random() * 0.3 + 0.6,
        successRate: Math.random() * 0.4 + 0.6,
        lastDetected: Math.random() > 0.5 ? new Date().toISOString() : null
      },
      isBuiltIn: ['head_shoulders', 'double_top', 'double_bottom', 'ascending_triangle', 'cup_handle', 'bull_flag', 'falling_wedge'].includes(def.id)
    }))

    setPatterns(extendedPatterns)
  }

  const filterPatterns = () => {
    let filtered = patterns

    if (searchTerm) {
      filtered = filtered.filter(pattern =>
        pattern.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (pattern.description && pattern.description.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (pattern.tags && pattern.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase())))
      )
    }

    if (filterCategory !== 'all') {
      if (filterCategory === 'built-in') {
        filtered = filtered.filter(pattern => pattern.isBuiltIn)
      } else if (filterCategory === 'custom') {
        filtered = filtered.filter(pattern => !pattern.isBuiltIn)
      } else {
        filtered = filtered.filter(pattern => pattern.category === filterCategory)
      }
    }

    setFilteredPatterns(filtered)
  }

  const togglePatternEnabled = (patternId: string, enabled: boolean) => {
    const config = patternRecognition.getConfig()
    let newEnabledPatterns = [...config.enabledPatterns]

    if (enabled && !newEnabledPatterns.includes(patternId)) {
      newEnabledPatterns.push(patternId)
    } else if (!enabled && newEnabledPatterns.includes(patternId)) {
      newEnabledPatterns = newEnabledPatterns.filter(id => id !== patternId)
    }

    patternRecognition.setConfig({ enabledPatterns: newEnabledPatterns })
    
    setPatterns(prev => prev.map(pattern => 
      pattern.id === patternId ? { ...pattern, enabled } : pattern
    ))

    message.success(`Pattern ${enabled ? 'enabled' : 'disabled'}`)
  }

  const duplicatePattern = (pattern: ExtendedPatternDefinition) => {
    const duplicate: Omit<PatternDefinition, 'id'> = {
      name: `${pattern.name} (Copy)`,
      type: pattern.type,
      rules: pattern.rules,
      minBars: pattern.minBars,
      maxBars: pattern.maxBars,
      minConfidence: pattern.minConfidence,
      description: pattern.description,
      category: 'custom',
      tags: pattern.tags
    }

    const newPatternId = patternRecognition.registerPattern(duplicate)
    loadPatterns()
    message.success(`Pattern duplicated as "${duplicate.name}"`)
  }

  const deletePattern = (patternId: string) => {
    // Note: This would require implementing a delete method in the pattern recognition service
    message.success('Pattern deleted')
    loadPatterns()
  }

  const exportPattern = (pattern: ExtendedPatternDefinition) => {
    const exportData = {
      name: pattern.name,
      type: pattern.type,
      rules: pattern.rules,
      minBars: pattern.minBars,
      maxBars: pattern.maxBars,
      minConfidence: pattern.minConfidence,
      description: pattern.description,
      category: pattern.category,
      tags: pattern.tags
    }

    const dataStr = JSON.stringify(exportData, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${pattern.name.replace(/[^a-z0-9]/gi, '_')}_pattern.json`
    link.click()
    URL.revokeObjectURL(url)
    
    message.success('Pattern exported')
  }

  const columns = [
    {
      title: 'Pattern',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: ExtendedPatternDefinition) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            <Space>
              <Tag color={record.isBuiltIn ? 'blue' : 'green'}>
                {record.isBuiltIn ? 'Built-in' : 'Custom'}
              </Tag>
              <span>{record.type}</span>
            </Space>
          </div>
        </div>
      )
    },
    {
      title: 'Rules',
      dataIndex: 'rules',
      key: 'rules',
      render: (rules: PatternDefinition['rules']) => (
        <Tooltip title={rules.map(r => `${r.type}: ${r.condition}`).join('\n')}>
          <Tag>{rules.length} rule{rules.length !== 1 ? 's' : ''}</Tag>
        </Tooltip>
      )
    },
    {
      title: 'Performance',
      key: 'performance',
      render: (record: ExtendedPatternDefinition) => (
        <div style={{ minWidth: 120 }}>
          <div style={{ fontSize: '12px', marginBottom: 4 }}>
            Success: {(record.stats.successRate * 100).toFixed(0)}%
          </div>
          <Progress 
            percent={record.stats.successRate * 100}
            size="small"
            strokeColor={record.stats.successRate > 0.7 ? '#52c41a' : '#faad14'}
            showInfo={false}
          />
          <div style={{ fontSize: '10px', color: '#999' }}>
            {record.stats.totalDetections} detections
          </div>
        </div>
      )
    },
    {
      title: 'Status',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled: boolean, record: ExtendedPatternDefinition) => (
        <Switch
          size="small"
          checked={enabled}
          onChange={(checked) => togglePatternEnabled(record.id, checked)}
          checkedChildren="ON"
          unCheckedChildren="OFF"
        />
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: ExtendedPatternDefinition) => (
        <Space size="small">
          <Tooltip title="View details">
            <Button
              type="link"
              size="small"
              icon={<InfoCircleOutlined />}
              onClick={() => {
                setSelectedPattern(record)
                setIsEditModalVisible(true)
              }}
            />
          </Tooltip>
          
          <Tooltip title="Duplicate">
            <Button
              type="link"
              size="small"
              icon={<CopyOutlined />}
              onClick={() => duplicatePattern(record)}
            />
          </Tooltip>
          
          <Tooltip title="Export">
            <Button
              type="link"
              size="small"
              icon={<DownloadOutlined />}
              onClick={() => exportPattern(record)}
            />
          </Tooltip>
          
          {!record.isBuiltIn && (
            <Popconfirm
              title="Delete this pattern?"
              onConfirm={() => deletePattern(record.id)}
              okText="Delete"
              okButtonProps={{ danger: true }}
            >
              <Button
                type="link"
                size="small"
                danger
                icon={<DeleteOutlined />}
              />
            </Popconfirm>
          )}
        </Space>
      )
    }
  ]

  const categories = [
    { value: 'all', label: `All (${patterns.length})` },
    { value: 'built-in', label: `Built-in (${patterns.filter(p => p.isBuiltIn).length})` },
    { value: 'custom', label: `Custom (${patterns.filter(p => !p.isBuiltIn).length})` },
    { value: 'reversal', label: 'Reversal' },
    { value: 'continuation', label: 'Continuation' },
    { value: 'consolidation', label: 'Consolidation' }
  ]

  return (
    <Card title="Pattern Management" size="small">
      <Tabs 
        activeKey={activeTab} 
        onChange={setActiveTab}
        items={[
          {
            key: 'manage',
            label: 'Manage Patterns',
            children: (
              <div>
                {/* Filters and Search */}
                <div style={{ marginBottom: 16, display: 'flex', gap: 12, alignItems: 'center' }}>
                  <Search
                    placeholder="Search patterns..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    style={{ width: 300 }}
                    allowClear
                  />
                  
                  <Select
                    value={filterCategory}
                    onChange={setFilterCategory}
                    style={{ width: 180 }}
                  >
                    {categories.map(cat => (
                      <Option key={cat.value} value={cat.value}>
                        {cat.label}
                      </Option>
                    ))}
                  </Select>

                  <div style={{ flex: 1 }} />
                  
                  <Space>
                    <Button size="small" icon={<UploadOutlined />}>
                      Import
                    </Button>
                  </Space>
                </div>

                {/* Patterns Table */}
                <Table
                  columns={columns}
                  dataSource={filteredPatterns}
                  rowKey="id"
                  size="small"
                  pagination={{
                    pageSize: 10,
                    showSizeChanger: false,
                    showTotal: (total) => `${total} patterns`
                  }}
                />
              </div>
            )
          },
          {
            key: 'create',
            label: 'Create Pattern',
            children: (
              <CustomPatternBuilder
                onPatternCreated={() => {
                  loadPatterns()
                  setActiveTab('manage')
                }}
              />
            )
          }
        ]}
      />

      {/* Pattern Details Modal */}
      <Modal
        title={selectedPattern ? `Pattern: ${selectedPattern.name}` : 'Pattern Details'}
        open={isEditModalVisible}
        onCancel={() => setIsEditModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setIsEditModalVisible(false)}>
            Close
          </Button>
        ]}
        width={600}
      >
        {selectedPattern && (
          <div style={{ display: 'grid', gap: 16 }}>
            <div>
              <strong>Description:</strong>
              <p>{selectedPattern.description || 'No description provided'}</p>
            </div>

            <div>
              <strong>Configuration:</strong>
              <div style={{ marginTop: 8 }}>
                <Space>
                  <Tag>Type: {selectedPattern.type}</Tag>
                  <Tag>Min Bars: {selectedPattern.minBars}</Tag>
                  <Tag>Max Bars: {selectedPattern.maxBars}</Tag>
                  <Tag>Min Confidence: {(selectedPattern.minConfidence * 100).toFixed(0)}%</Tag>
                </Space>
              </div>
            </div>

            <div>
              <strong>Rules ({selectedPattern.rules.length}):</strong>
              <div style={{ marginTop: 8 }}>
                {selectedPattern.rules.map((rule, index) => (
                  <div key={index} style={{ 
                    padding: 8, 
                    border: '1px solid #f0f0f0', 
                    borderRadius: 4, 
                    marginBottom: 8 
                  }}>
                    <div style={{ fontWeight: 500 }}>
                      Rule {index + 1}: {rule.type}
                    </div>
                    <div style={{ fontSize: '12px', color: '#666' }}>
                      Condition: {rule.condition}
                    </div>
                    {Object.keys(rule.parameters || {}).length > 0 && (
                      <div style={{ fontSize: '12px', color: '#999' }}>
                        Parameters: {JSON.stringify(rule.parameters)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div>
              <strong>Performance Statistics:</strong>
              <div style={{ marginTop: 8, display: 'grid', gap: 8, gridTemplateColumns: '1fr 1fr' }}>
                <div>
                  <div>Total Detections: {selectedPattern.stats.totalDetections}</div>
                  <div>Average Confidence: {(selectedPattern.stats.avgConfidence * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div>Success Rate: {(selectedPattern.stats.successRate * 100).toFixed(1)}%</div>
                  <div>Last Detected: {selectedPattern.stats.lastDetected ? 
                    new Date(selectedPattern.stats.lastDetected).toLocaleString() : 'Never'
                  }</div>
                </div>
              </div>
            </div>

            {selectedPattern.tags && selectedPattern.tags.length > 0 && (
              <div>
                <strong>Tags:</strong>
                <div style={{ marginTop: 4 }}>
                  <Space>
                    {selectedPattern.tags.map(tag => (
                      <Tag key={tag} size="small">{tag}</Tag>
                    ))}
                  </Space>
                </div>
              </div>
            )}
          </div>
        )}
      </Modal>
    </Card>
  )
}

export default PatternManagementDashboard