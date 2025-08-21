import React from 'react'
import { Modal, Space, Typography, Divider } from 'antd'
import { QuestionCircleOutlined } from '@ant-design/icons'

const { Text, Title } = Typography

interface KeyboardShortcutsHelpProps {
  visible: boolean
  onClose: () => void
}

interface ShortcutItem {
  keys: string[]
  description: string
  category: string
}

const shortcuts: ShortcutItem[] = [
  // Global shortcuts
  { keys: ['Cmd/Ctrl', 'K'], description: 'Focus search input', category: 'Global' },
  { keys: ['/'], description: 'Focus search input (alternative)', category: 'Global' },
  { keys: ['Cmd/Ctrl', 'Shift', 'F'], description: 'Toggle filters', category: 'Global' },
  { keys: ['Cmd/Ctrl', 'Shift', 'S'], description: 'Toggle ranking settings', category: 'Global' },
  
  // Search shortcuts
  { keys: ['↑', '↓'], description: 'Navigate search results', category: 'Search' },
  { keys: ['Tab', 'Shift+Tab'], description: 'Navigate search results (alternative)', category: 'Search' },
  { keys: ['Enter'], description: 'Select highlighted instrument', category: 'Search' },
  { keys: ['Escape'], description: 'Clear search and lose focus', category: 'Search' },
  { keys: ['1-9'], description: 'Quick select first 9 results', category: 'Search' },
]

const KeyboardShortcut: React.FC<{ keys: string[]; description: string }> = ({ keys, description }) => (
  <div className="flex justify-between items-center py-2">
    <div className="flex items-center space-x-2">
      {keys.map((key, index) => (
        <React.Fragment key={index}>
          {index > 0 && <Text type="secondary">+</Text>}
          <kbd className="px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded-lg">
            {key}
          </kbd>
        </React.Fragment>
      ))}
    </div>
    <Text className="ml-4 text-gray-600">{description}</Text>
  </div>
)

export const KeyboardShortcutsHelp: React.FC<KeyboardShortcutsHelpProps> = ({ visible, onClose }) => {
  const groupedShortcuts = shortcuts.reduce((acc, shortcut) => {
    if (!acc[shortcut.category]) {
      acc[shortcut.category] = []
    }
    acc[shortcut.category].push(shortcut)
    return acc
  }, {} as Record<string, ShortcutItem[]>)

  return (
    <Modal
      title={
        <Space>
          <QuestionCircleOutlined />
          Keyboard Shortcuts
        </Space>
      }
      open={visible}
      onCancel={onClose}
      footer={null}
      width={600}
    >
      <div className="space-y-6">
        <Text type="secondary">
          Use these keyboard shortcuts to navigate and interact with the instrument search more efficiently.
        </Text>

        {Object.entries(groupedShortcuts).map(([category, categoryShortcuts]) => (
          <div key={category}>
            <Title level={5} className="mb-3">
              {category} Shortcuts
            </Title>
            <div className="space-y-1">
              {categoryShortcuts.map((shortcut, index) => (
                <KeyboardShortcut
                  key={`${category}-${index}`}
                  keys={shortcut.keys}
                  description={shortcut.description}
                />
              ))}
            </div>
            {category !== 'Search' && <Divider />}
          </div>
        ))}

        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <Title level={5} className="mb-2">
            Pro Tips
          </Title>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• Use number keys (1-9) for instant selection of search results</li>
            <li>• Arrow keys and Tab both work for navigation - use what feels natural</li>
            <li>• Press <kbd className="px-1 py-0.5 text-xs bg-gray-200 rounded">Escape</kbd> to clear search and return to overview</li>
            <li>• Combine shortcuts: <kbd className="px-1 py-0.5 text-xs bg-gray-200 rounded">/</kbd> to search, then numbers to select</li>
          </ul>
        </div>
      </div>
    </Modal>
  )
}

export default KeyboardShortcutsHelp