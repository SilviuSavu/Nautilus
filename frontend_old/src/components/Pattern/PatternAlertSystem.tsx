/**
 * Pattern Alert System Component
 * Manages pattern-based alerts and notifications
 */

import React, { useState, useEffect, useCallback } from 'react'
import { 
  Card, 
  List, 
  Button, 
  Space, 
  Tag, 
  Modal, 
  Form, 
  Input, 
  Select, 
  InputNumber,
  Switch,
  Badge,
  Tooltip,
  message,
  Divider,
  Tabs,
  Alert,
  Progress
} from 'antd'
import { 
  BellOutlined, 
  PlusOutlined, 
  DeleteOutlined,
  EditOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  SettingOutlined,
  NotificationOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined
} from '@ant-design/icons'
import { PatternAlert } from '../../services/patternRecognition'
import { patternRecognition } from '../../services/patternRecognition'

const { Option } = Select
const { TextArea } = Input

interface AlertRule {
  id: string
  name: string
  description: string
  patternIds: string[]
  instruments: string[]
  timeframes: string[]
  minConfidence: number
  alertTypes: ('formation' | 'completion' | 'breakout')[]
  enabled: boolean
  priority: 'low' | 'medium' | 'high' | 'critical'
  notifications: {
    sound: boolean
    email: boolean
    push: boolean
    popup: boolean
  }
  conditions: {
    volumeConfirmation?: boolean
    priceTargets?: number[]
    timeOfDay?: { start: string; end: string }
  }
  createdAt: string
  triggeredCount: number
  lastTriggered?: string
}

interface AlertNotification extends PatternAlert {
  ruleName: string
  timestamp: string
  acknowledged: boolean
  priority: AlertRule['priority']
}

export const PatternAlertSystem: React.FC = () => {
  const [alertRules, setAlertRules] = useState<AlertRule[]>([])
  const [notifications, setNotifications] = useState<AlertNotification[]>([])
  const [isRuleModalVisible, setIsRuleModalVisible] = useState(false)
  const [editingRule, setEditingRule] = useState<AlertRule | null>(null)
  const [form] = Form.useForm()
  const [activeTab, setActiveTab] = useState('rules')

  useEffect(() => {
    loadAlertRules()
    loadNotifications()
    startAlertMonitoring()
  }, [])

  const loadAlertRules = () => {
    try {
      const stored = localStorage.getItem('patternAlertRules')
      if (stored) {
        setAlertRules(JSON.parse(stored))
      }
    } catch (error) {
      console.error('Failed to load alert rules:', error)
    }
  }

  const saveAlertRules = (rules: AlertRule[]) => {
    try {
      localStorage.setItem('patternAlertRules', JSON.stringify(rules))
      setAlertRules(rules)
    } catch (error) {
      console.error('Failed to save alert rules:', error)
    }
  }

  const loadNotifications = () => {
    try {
      const stored = localStorage.getItem('patternAlertNotifications')
      if (stored) {
        setNotifications(JSON.parse(stored))
      }
    } catch (error) {
      console.error('Failed to load notifications:', error)
    }
  }

  const saveNotifications = (notifications: AlertNotification[]) => {
    try {
      localStorage.setItem('patternAlertNotifications', JSON.stringify(notifications))
      setNotifications(notifications)
    } catch (error) {
      console.error('Failed to save notifications:', error)
    }
  }

  const startAlertMonitoring = useCallback(() => {
    // Simulate real-time pattern monitoring
    const monitoringInterval = setInterval(() => {
      checkForPatternAlerts()
    }, 30000) // Check every 30 seconds

    return () => clearInterval(monitoringInterval)
  }, [alertRules])

  const checkForPatternAlerts = async () => {
    const enabledRules = alertRules.filter(rule => rule.enabled)
    
    for (const rule of enabledRules) {
      // In a real implementation, this would check actual market data
      // For now, we'll simulate pattern detection
      if (Math.random() < 0.01) { // 1% chance to trigger an alert
        const mockAlert = createMockAlert(rule)
        if (mockAlert) {
          triggerAlert(mockAlert, rule)
        }
      }
    }
  }

  const createMockAlert = (rule: AlertRule): PatternAlert | null => {
    if (rule.patternIds.length === 0) return null

    const randomPatternId = rule.patternIds[Math.floor(Math.random() * rule.patternIds.length)]
    const alertTypes = rule.alertTypes.length > 0 ? rule.alertTypes : ['formation']
    const randomAlertType = alertTypes[Math.floor(Math.random() * alertTypes.length)]

    return {
      id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      patternId: randomPatternId,
      triggeredAt: new Date().toISOString(),
      alertType: randomAlertType,
      confidence: Math.random() * 0.3 + 0.7,
      message: `${randomAlertType} detected for pattern ${randomPatternId}`
    }
  }

  const triggerAlert = (alert: PatternAlert, rule: AlertRule) => {
    const notification: AlertNotification = {
      ...alert,
      ruleName: rule.name,
      timestamp: new Date().toISOString(),
      acknowledged: false,
      priority: rule.priority
    }

    const updatedNotifications = [notification, ...notifications]
    saveNotifications(updatedNotifications)

    // Update rule trigger count
    const updatedRules = alertRules.map(r => 
      r.id === rule.id ? {
        ...r,
        triggeredCount: r.triggeredCount + 1,
        lastTriggered: new Date().toISOString()
      } : r
    )
    saveAlertRules(updatedRules)

    // Show notification based on rule settings
    if (rule.notifications.popup) {
      showAlertPopup(notification)
    }

    if (rule.notifications.sound) {
      playAlertSound(rule.priority)
    }

    message.info(`Pattern alert: ${notification.message}`)
  }

  const showAlertPopup = (notification: AlertNotification) => {
    Modal.info({
      title: 'Pattern Alert',
      content: (
        <div>
          <div><strong>Rule:</strong> {notification.ruleName}</div>
          <div><strong>Type:</strong> {notification.alertType}</div>
          <div><strong>Confidence:</strong> {(notification.confidence * 100).toFixed(1)}%</div>
          <div><strong>Message:</strong> {notification.message}</div>
        </div>
      )
    })
  }

  const playAlertSound = (priority: AlertRule['priority']) => {
    // In a real implementation, this would play different sounds based on priority
    console.log(`Playing ${priority} priority alert sound`)
  }

  const createOrUpdateRule = async () => {
    try {
      const values = await form.validateFields()
      
      const rule: AlertRule = {
        id: editingRule?.id || `rule_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        name: values.name,
        description: values.description || '',
        patternIds: values.patternIds || [],
        instruments: values.instruments || [],
        timeframes: values.timeframes || [],
        minConfidence: values.minConfidence || 0.6,
        alertTypes: values.alertTypes || ['formation'],
        enabled: values.enabled !== false,
        priority: values.priority || 'medium',
        notifications: {
          sound: values.notificationSound !== false,
          email: values.notificationEmail || false,
          push: values.notificationPush || false,
          popup: values.notificationPopup !== false
        },
        conditions: {
          volumeConfirmation: values.volumeConfirmation || false,
          priceTargets: values.priceTargets || [],
          timeOfDay: values.timeOfDay
        },
        createdAt: editingRule?.createdAt || new Date().toISOString(),
        triggeredCount: editingRule?.triggeredCount || 0,
        lastTriggered: editingRule?.lastTriggered
      }

      const updatedRules = editingRule 
        ? alertRules.map(r => r.id === editingRule.id ? rule : r)
        : [...alertRules, rule]

      saveAlertRules(updatedRules)
      
      setIsRuleModalVisible(false)
      setEditingRule(null)
      form.resetFields()
      
      message.success(`Alert rule ${editingRule ? 'updated' : 'created'} successfully`)
    } catch (error) {
      message.error('Failed to save alert rule')
    }
  }

  const deleteRule = (ruleId: string) => {
    const updatedRules = alertRules.filter(rule => rule.id !== ruleId)
    saveAlertRules(updatedRules)
    message.success('Alert rule deleted')
  }

  const toggleRuleEnabled = (ruleId: string, enabled: boolean) => {
    const updatedRules = alertRules.map(rule => 
      rule.id === ruleId ? { ...rule, enabled } : rule
    )
    saveAlertRules(updatedRules)
    message.success(`Rule ${enabled ? 'enabled' : 'disabled'}`)
  }

  const acknowledgeNotification = (notificationId: string) => {
    const updatedNotifications = notifications.map(notification =>
      notification.id === notificationId ? { ...notification, acknowledged: true } : notification
    )
    saveNotifications(updatedNotifications)
  }

  const clearAllNotifications = () => {
    saveNotifications([])
    message.success('All notifications cleared')
  }

  const getPatternOptions = () => {
    const patterns = patternRecognition.getPatternDefinitions()
    return patterns.map(pattern => ({
      value: pattern.id,
      label: pattern.name
    }))
  }

  const getPriorityColor = (priority: AlertRule['priority']) => {
    const colors = {
      low: 'blue',
      medium: 'orange',
      high: 'red',
      critical: 'purple'
    }
    return colors[priority]
  }

  const unacknowledgedCount = notifications.filter(n => !n.acknowledged).length

  return (
    <Card 
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Pattern Alert System</span>
          <Badge count={unacknowledgedCount} showZero>
            <BellOutlined style={{ fontSize: 18 }} />
          </Badge>
        </div>
      } 
      size="small"
    >
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'rules',
            label: `Rules (${alertRules.length})`,
            children: (
              <div>
                {/* Create Rule Button */}
                <div style={{ marginBottom: 16 }}>
                  <Button
                    type="primary"
                    icon={<PlusOutlined />}
                    onClick={() => {
                      setEditingRule(null)
                      setIsRuleModalVisible(true)
                    }}
                  >
                    Create Alert Rule
                  </Button>
                </div>

                {/* Alert Rules List */}
                <List
                  dataSource={alertRules}
                  renderItem={(rule) => (
                    <List.Item
                      actions={[
                        <Tooltip title={rule.enabled ? 'Disable' : 'Enable'}>
                          <Button
                            type="link"
                            size="small"
                            icon={rule.enabled ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                            onClick={() => toggleRuleEnabled(rule.id, !rule.enabled)}
                          />
                        </Tooltip>,
                        <Tooltip title="Edit rule">
                          <Button
                            type="link"
                            size="small"
                            icon={<EditOutlined />}
                            onClick={() => {
                              setEditingRule(rule)
                              form.setFieldsValue({
                                name: rule.name,
                                description: rule.description,
                                patternIds: rule.patternIds,
                                instruments: rule.instruments,
                                timeframes: rule.timeframes,
                                minConfidence: rule.minConfidence,
                                alertTypes: rule.alertTypes,
                                enabled: rule.enabled,
                                priority: rule.priority,
                                notificationSound: rule.notifications.sound,
                                notificationEmail: rule.notifications.email,
                                notificationPush: rule.notifications.push,
                                notificationPopup: rule.notifications.popup
                              })
                              setIsRuleModalVisible(true)
                            }}
                          />
                        </Tooltip>,
                        <Tooltip title="Delete rule">
                          <Button
                            type="link"
                            size="small"
                            danger
                            icon={<DeleteOutlined />}
                            onClick={() => {
                              Modal.confirm({
                                title: 'Delete Alert Rule',
                                content: `Are you sure you want to delete "${rule.name}"?`,
                                onOk: () => deleteRule(rule.id)
                              })
                            }}
                          />
                        </Tooltip>
                      ]}
                    >
                      <List.Item.Meta
                        title={
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span>{rule.name}</span>
                            <Tag color={getPriorityColor(rule.priority)}>{rule.priority}</Tag>
                            {!rule.enabled && <Tag color="red">Disabled</Tag>}
                          </div>
                        }
                        description={
                          <div>
                            <div style={{ marginBottom: 4 }}>{rule.description}</div>
                            <div style={{ fontSize: '12px', color: '#999' }}>
                              <Space>
                                <span>{rule.patternIds.length} pattern{rule.patternIds.length !== 1 ? 's' : ''}</span>
                                <span>Min confidence: {(rule.minConfidence * 100).toFixed(0)}%</span>
                                <span>Triggered: {rule.triggeredCount} times</span>
                              </Space>
                            </div>
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />

                {alertRules.length === 0 && (
                  <div style={{ 
                    textAlign: 'center', 
                    padding: 40, 
                    color: '#999',
                    border: '1px dashed #d9d9d9',
                    borderRadius: 6
                  }}>
                    No alert rules configured. Create your first rule to start monitoring patterns.
                  </div>
                )}
              </div>
            )
          },
          {
            key: 'notifications',
            label: (
              <Badge count={unacknowledgedCount} size="small">
                <span>Notifications</span>
              </Badge>
            ),
            children: (
              <div>
                {/* Notifications Header */}
                <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <strong>{notifications.length} notifications</strong>
                    {unacknowledgedCount > 0 && (
                      <span style={{ marginLeft: 8, color: '#ff4d4f' }}>
                        ({unacknowledgedCount} unread)
                      </span>
                    )}
                  </div>
                  
                  {notifications.length > 0 && (
                    <Button size="small" onClick={clearAllNotifications}>
                      Clear All
                    </Button>
                  )}
                </div>

                {/* Notifications List */}
                <List
                  dataSource={notifications}
                  renderItem={(notification) => (
                    <List.Item
                      style={{ 
                        backgroundColor: notification.acknowledged ? '#fafafa' : '#fff',
                        border: `1px solid ${notification.acknowledged ? '#f0f0f0' : '#1890ff'}`,
                        borderRadius: 6,
                        marginBottom: 8,
                        padding: '12px 16px'
                      }}
                      actions={[
                        !notification.acknowledged && (
                          <Button
                            type="link"
                            size="small"
                            onClick={() => acknowledgeNotification(notification.id)}
                          >
                            Acknowledge
                          </Button>
                        )
                      ].filter(Boolean)}
                    >
                      <List.Item.Meta
                        avatar={
                          <div style={{ fontSize: 16 }}>
                            {notification.alertType === 'formation' && '🔵'}
                            {notification.alertType === 'completion' && '🟢'}
                            {notification.alertType === 'breakout' && '🔴'}
                          </div>
                        }
                        title={
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span>{notification.ruleName}</span>
                            <Tag color={getPriorityColor(notification.priority)}>
                              {notification.priority}
                            </Tag>
                            <Tag>{notification.alertType}</Tag>
                          </div>
                        }
                        description={
                          <div>
                            <div style={{ marginBottom: 4 }}>{notification.message}</div>
                            <div style={{ fontSize: '12px', color: '#999' }}>
                              <Space>
                                <span>Confidence: {(notification.confidence * 100).toFixed(1)}%</span>
                                <span>{new Date(notification.timestamp).toLocaleString()}</span>
                              </Space>
                            </div>
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />

                {notifications.length === 0 && (
                  <div style={{ 
                    textAlign: 'center', 
                    padding: 40, 
                    color: '#999',
                    border: '1px dashed #d9d9d9',
                    borderRadius: 6
                  }}>
                    No notifications yet. Alerts will appear here when patterns are detected.
                  </div>
                )}
              </div>
            )
          }
        ]}
      />

      {/* Create/Edit Rule Modal */}
      <Modal
        title={editingRule ? 'Edit Alert Rule' : 'Create Alert Rule'}
        open={isRuleModalVisible}
        onOk={createOrUpdateRule}
        onCancel={() => {
          setIsRuleModalVisible(false)
          setEditingRule(null)
          form.resetFields()
        }}
        width={600}
        okText={editingRule ? 'Update' : 'Create'}
      >
        <Form form={form} layout="vertical" size="small">
          <div style={{ display: 'grid', gap: 12, gridTemplateColumns: '1fr 1fr' }}>
            <Form.Item
              name="name"
              label="Rule Name"
              rules={[{ required: true, message: 'Please enter rule name' }]}
              style={{ gridColumn: '1 / -1' }}
            >
              <Input placeholder="e.g., AAPL Head & Shoulders Alert" />
            </Form.Item>

            <Form.Item name="description" label="Description" style={{ gridColumn: '1 / -1' }}>
              <TextArea rows={2} placeholder="Optional description of what this rule monitors" />
            </Form.Item>

            <Form.Item
              name="patternIds"
              label="Patterns to Monitor"
              rules={[{ required: true, message: 'Please select at least one pattern' }]}
            >
              <Select
                mode="multiple"
                placeholder="Select patterns"
                options={getPatternOptions()}
              />
            </Form.Item>

            <Form.Item name="instruments" label="Instruments">
              <Select
                mode="tags"
                placeholder="e.g., AAPL, MSFT (leave empty for all)"
                style={{ width: '100%' }}
              />
            </Form.Item>

            <Form.Item name="timeframes" label="Timeframes">
              <Select mode="multiple" placeholder="Select timeframes">
                <Option value="1m">1 minute</Option>
                <Option value="5m">5 minutes</Option>
                <Option value="15m">15 minutes</Option>
                <Option value="1h">1 hour</Option>
                <Option value="4h">4 hours</Option>
                <Option value="1d">1 day</Option>
              </Select>
            </Form.Item>

            <Form.Item name="alertTypes" label="Alert Types">
              <Select mode="multiple" placeholder="Select alert types">
                <Option value="formation">Formation</Option>
                <Option value="completion">Completion</Option>
                <Option value="breakout">Breakout</Option>
              </Select>
            </Form.Item>

            <Form.Item name="minConfidence" label="Minimum Confidence">
              <InputNumber
                min={0.1}
                max={1}
                step={0.05}
                formatter={value => `${(Number(value) * 100).toFixed(0)}%`}
                parser={value => Number(value!.replace('%', '')) / 100}
                style={{ width: '100%' }}
              />
            </Form.Item>

            <Form.Item name="priority" label="Priority">
              <Select>
                <Option value="low">Low</Option>
                <Option value="medium">Medium</Option>
                <Option value="high">High</Option>
                <Option value="critical">Critical</Option>
              </Select>
            </Form.Item>
          </div>

          <Divider>Notification Settings</Divider>

          <div style={{ display: 'grid', gap: 8, gridTemplateColumns: '1fr 1fr' }}>
            <Form.Item name="notificationPopup" valuePropName="checked">
              <Switch /> <span style={{ marginLeft: 8 }}>Popup Alert</span>
            </Form.Item>

            <Form.Item name="notificationSound" valuePropName="checked">
              <Switch /> <span style={{ marginLeft: 8 }}>Sound Alert</span>
            </Form.Item>

            <Form.Item name="notificationEmail" valuePropName="checked">
              <Switch /> <span style={{ marginLeft: 8 }}>Email Alert</span>
            </Form.Item>

            <Form.Item name="notificationPush" valuePropName="checked">
              <Switch /> <span style={{ marginLeft: 8 }}>Push Notification</span>
            </Form.Item>
          </div>

          <Form.Item name="enabled" valuePropName="checked" style={{ marginBottom: 0 }}>
            <Switch /> <span style={{ marginLeft: 8 }}>Enable this rule</span>
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  )
}

export default PatternAlertSystem