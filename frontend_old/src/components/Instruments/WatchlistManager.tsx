import React, { useState } from 'react'
import { 
  Card, 
  List, 
  Space, 
  Button, 
  Dropdown, 
  Modal, 
  Input, 
  Form, 
  message, 
  Popconfirm,
  Tag,
  Tooltip,
  Badge,
  Empty,
  Typography
} from 'antd'
import { 
  PlusOutlined, 
  MoreOutlined, 
  DeleteOutlined, 
  EditOutlined,
  StarOutlined,
  StarFilled,
  ExportOutlined,
  ImportOutlined,
  FolderOutlined,
  HeartFilled,
  DragOutlined,
  CloseOutlined
} from '@ant-design/icons'
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core'
import { arrayMove, SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy } from '@dnd-kit/sortable'
import { useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useInstrumentStore } from './hooks/useInstrumentStore'
import { VenueStatusIndicator } from './VenueStatusIndicator'
import { WatchlistImportExport } from './WatchlistImportExport'
import { WatchlistPrice } from './RealtimePriceDisplay'
import { Watchlist, WatchlistItem, Instrument } from './types/instrumentTypes'

const { Title, Text } = Typography

interface WatchlistManagerProps {
  className?: string
  onInstrumentSelect?: (instrument: Instrument) => void
  showCreateButton?: boolean
  compactMode?: boolean
}

interface WatchlistItemComponentProps {
  item: WatchlistItem
  onRemove: (instrumentId: string) => void
  onAddNote: (instrumentId: string, note: string) => void
  onInstrumentClick: (instrument: Instrument) => void
  compact?: boolean
}

interface SortableWatchlistItemProps extends WatchlistItemComponentProps {
  id: string
}

const SortableWatchlistItem: React.FC<SortableWatchlistItemProps> = ({ 
  id, 
  item, 
  onRemove, 
  onAddNote, 
  onInstrumentClick,
  compact = false 
}) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  }

  return (
    <div ref={setNodeRef} style={style} {...attributes}>
      <WatchlistItemComponent
        item={item}
        onRemove={onRemove}
        onAddNote={onAddNote}
        onInstrumentClick={onInstrumentClick}
        compact={compact}
        dragHandle={
          <Button 
            type="text" 
            size="small" 
            icon={<DragOutlined />} 
            style={{ cursor: 'grab' }}
            {...listeners}
          />
        }
      />
    </div>
  )
}

const WatchlistItemComponent: React.FC<WatchlistItemComponentProps & { dragHandle?: React.ReactNode }> = ({
  item,
  onRemove,
  onAddNote,
  onInstrumentClick,
  compact = false,
  dragHandle
}) => {
  const [noteModalVisible, setNoteModalVisible] = useState(false)
  const [noteForm] = Form.useForm()
  const { venueStatus, isFavorite, addToFavorites, removeFromFavorites } = useInstrumentStore()

  const handleNoteSubmit = async () => {
    try {
      const values = await noteForm.validateFields()
      onAddNote(item.instrument.id, values.note)
      setNoteModalVisible(false)
      noteForm.resetFields()
      message.success('Note added successfully')
    } catch (error) {
      console.error('Note validation failed:', error)
    }
  }

  const handleFavoriteToggle = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (isFavorite(item.instrument.id)) {
      removeFromFavorites(item.instrument.id)
    } else {
      addToFavorites(item.instrument)
    }
  }

  const getAssetClassColor = (assetClass: string) => {
    const colorMap: Record<string, string> = {
      'STK': 'blue',
      'CASH': 'green',
      'FUT': 'orange',
      'IND': 'purple',
      'OPT': 'red',
      'BOND': 'cyan',
      'CRYPTO': 'gold'
    }
    return colorMap[assetClass] || 'default'
  }

  if (compact) {
    return (
      <List.Item
        onClick={() => onInstrumentClick(item.instrument)}
        style={{ cursor: 'pointer', padding: '8px 12px' }}
        actions={[
          dragHandle,
          <Button
            key="favorite"
            type="text"
            size="small"
            icon={isFavorite(item.instrument.id) ? <HeartFilled /> : <StarOutlined />}
            onClick={handleFavoriteToggle}
            className={isFavorite(item.instrument.id) ? "text-red-500" : "text-gray-400"}
          />,
          <Button
            key="remove"
            type="text"
            size="small"
            icon={<CloseOutlined />}
            onClick={(e) => {
              e.stopPropagation()
              onRemove(item.instrument.id)
            }}
            danger
          />
        ]}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
          <Space>
            <Tag color={getAssetClassColor(item.instrument.assetClass)} size="small">
              {item.instrument.assetClass}
            </Tag>
            <strong style={{ fontSize: '13px' }}>{item.instrument.symbol}</strong>
            <VenueStatusIndicator
              venue={item.instrument.venue}
              status={venueStatus[item.instrument.venue]}
              showText={false}
              size="small"
            />
          </Space>
          <WatchlistPrice instrument={item.instrument} />
        </div>
      </List.Item>
    )
  }

  return (
    <>
      <List.Item
        onClick={() => onInstrumentClick(item.instrument)}
        style={{ cursor: 'pointer', padding: '12px 16px' }}
        actions={[
          dragHandle,
          <Dropdown
            key="actions"
            menu={{
              items: [
                {
                  key: 'note',
                  label: 'Add Note',
                  icon: <EditOutlined />,
                  onClick: (e) => {
                    e.domEvent.stopPropagation()
                    setNoteModalVisible(true)
                  }
                },
                {
                  key: 'favorite',
                  label: isFavorite(item.instrument.id) ? 'Remove from Favorites' : 'Add to Favorites',
                  icon: isFavorite(item.instrument.id) ? <StarFilled /> : <StarOutlined />,
                  onClick: (e) => {
                    e.domEvent.stopPropagation()
                    handleFavoriteToggle(e.domEvent)
                  }
                },
                { type: 'divider' },
                {
                  key: 'remove',
                  label: 'Remove from Watchlist',
                  icon: <DeleteOutlined />,
                  danger: true,
                  onClick: (e) => {
                    e.domEvent.stopPropagation()
                    onRemove(item.instrument.id)
                  }
                }
              ]
            }}
            trigger={['click']}
          >
            <Button
              type="text"
              size="small"
              icon={<MoreOutlined />}
              onClick={(e) => e.stopPropagation()}
            />
          </Dropdown>
        ]}
      >
        <List.Item.Meta
          title={
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Space>
                <Tag color={getAssetClassColor(item.instrument.assetClass)}>
                  {item.instrument.assetClass}
                </Tag>
                <strong>{item.instrument.symbol}</strong>
                <VenueStatusIndicator
                  venue={item.instrument.venue}
                  status={venueStatus[item.instrument.venue]}
                  showText={true}
                  size="small"
                />
                {isFavorite(item.instrument.id) && (
                  <HeartFilled className="text-red-500" />
                )}
              </Space>
              <WatchlistPrice instrument={item.instrument} />
            </div>
          }
          description={
            <div>
              <div style={{ marginBottom: '4px' }}>
                {item.instrument.name}
              </div>
              <Space size="small" style={{ fontSize: '12px', color: '#666' }}>
                <span>{item.instrument.currency}</span>
                <span>•</span>
                <span>Added: {new Date(item.addedAt).toLocaleDateString()}</span>
                {item.notes && (
                  <>
                    <span>•</span>
                    <Tooltip title={item.notes}>
                      <Text type="secondary" ellipsis style={{ maxWidth: '200px' }}>
                        📝 {item.notes}
                      </Text>
                    </Tooltip>
                  </>
                )}
              </Space>
            </div>
          }
        />
      </List.Item>

      <Modal
        title="Add Note"
        open={noteModalVisible}
        onOk={handleNoteSubmit}
        onCancel={() => {
          setNoteModalVisible(false)
          noteForm.resetFields()
        }}
        okText="Save Note"
      >
        <Form form={noteForm} layout="vertical">
          <Form.Item
            name="note"
            label="Note"
            rules={[{ required: true, message: 'Please enter a note' }]}
          >
            <Input.TextArea
              rows={3}
              placeholder="Add a note for this instrument..."
              defaultValue={item.notes}
            />
          </Form.Item>
        </Form>
      </Modal>
    </>
  )
}

export const WatchlistManager: React.FC<WatchlistManagerProps> = ({
  className,
  onInstrumentSelect,
  showCreateButton = true,
  compactMode = false
}) => {
  const [selectedWatchlistId, setSelectedWatchlistId] = useState<string>('default')
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [editModalVisible, setEditModalVisible] = useState(false)
  const [importExportVisible, setImportExportVisible] = useState(false)
  const [importExportMode, setImportExportMode] = useState<'import' | 'export'>('export')
  const [createForm] = Form.useForm()
  const [editForm] = Form.useForm()

  const {
    watchlists,
    createWatchlist,
    deleteWatchlist,
    addToWatchlist,
    removeFromWatchlist,
    getWatchlist,
    updateWatchlistItem
  } = useInstrumentStore()

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  )

  const selectedWatchlist = getWatchlist(selectedWatchlistId)

  const handleCreateWatchlist = async () => {
    try {
      const values = await createForm.validateFields()
      const newWatchlist = createWatchlist(values.name, values.description)
      setSelectedWatchlistId(newWatchlist.id)
      setCreateModalVisible(false)
      createForm.resetFields()
      message.success('Watchlist created successfully')
    } catch (error) {
      console.error('Watchlist creation failed:', error)
    }
  }

  const handleDeleteWatchlist = (watchlistId: string) => {
    if (watchlistId === 'default') {
      message.error('Cannot delete the default watchlist')
      return
    }
    deleteWatchlist(watchlistId)
    setSelectedWatchlistId('default')
    message.success('Watchlist deleted successfully')
  }

  const handleRemoveFromWatchlist = (instrumentId: string) => {
    removeFromWatchlist(selectedWatchlistId, instrumentId)
    message.success('Instrument removed from watchlist')
  }

  const handleAddNote = (instrumentId: string, note: string) => {
    updateWatchlistItem(selectedWatchlistId, instrumentId, { notes: note })
  }

  const handleDragEnd = (event: any) => {
    const { active, over } = event

    if (active.id !== over.id && selectedWatchlist) {
      const oldIndex = selectedWatchlist.items.findIndex(item => item.instrument.id === active.id)
      const newIndex = selectedWatchlist.items.findIndex(item => item.instrument.id === over.id)
      
      const newItems = arrayMove(selectedWatchlist.items, oldIndex, newIndex)
      // Update the watchlist order
      // This would require an additional store method to reorder items
      console.log('Reordering items:', { oldIndex, newIndex, newItems })
    }
  }

  const watchlistTabs = watchlists.map(watchlist => ({
    key: watchlist.id,
    label: (
      <Space>
        <FolderOutlined />
        {watchlist.name}
        <Badge count={watchlist.items.length} size="small" />
      </Space>
    )
  }))

  if (!selectedWatchlist) {
    return (
      <Card className={className}>
        <Empty description="Watchlist not found" />
      </Card>
    )
  }

  return (
    <div className={className}>
      <Card
        title={
          <Space>
            <FolderOutlined />
            Watchlists
            <Badge count={selectedWatchlist.items.length} />
          </Space>
        }
        extra={
          <Space>
            {showCreateButton && (
              <Button
                type="primary"
                size="small"
                icon={<PlusOutlined />}
                onClick={() => setCreateModalVisible(true)}
              >
                New Watchlist
              </Button>
            )}
            <Dropdown
              menu={{
                items: [
                  {
                    key: 'export',
                    label: 'Export Watchlist',
                    icon: <ExportOutlined />,
                    onClick: () => {
                      setImportExportMode('export')
                      setImportExportVisible(true)
                    }
                  },
                  {
                    key: 'import',
                    label: 'Import Watchlist',
                    icon: <ImportOutlined />,
                    onClick: () => {
                      setImportExportMode('import')
                      setImportExportVisible(true)
                    }
                  }
                ]
              }}
            >
              <Button size="small" icon={<MoreOutlined />} />
            </Dropdown>
          </Space>
        }
      >
        {/* Watchlist Selector */}
        <div style={{ marginBottom: '16px' }}>
          <Space wrap>
            {watchlists.map(watchlist => (
              <Button
                key={watchlist.id}
                type={selectedWatchlistId === watchlist.id ? 'primary' : 'default'}
                size="small"
                onClick={() => setSelectedWatchlistId(watchlist.id)}
              >
                <Space>
                  {watchlist.name}
                  <Badge count={watchlist.items.length} size="small" />
                </Space>
              </Button>
            ))}
          </Space>
        </div>

        {/* Watchlist Items */}
        {selectedWatchlist.items.length === 0 ? (
          <Empty
            description="No instruments in this watchlist"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          >
            <Text type="secondary">
              Add instruments from the search results to build your watchlist
            </Text>
          </Empty>
        ) : (
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext
              items={selectedWatchlist.items.map(item => item.instrument.id)}
              strategy={verticalListSortingStrategy}
            >
              <List
                dataSource={selectedWatchlist.items}
                renderItem={(item) => (
                  <SortableWatchlistItem
                    key={item.instrument.id}
                    id={item.instrument.id}
                    item={item}
                    onRemove={handleRemoveFromWatchlist}
                    onAddNote={handleAddNote}
                    onInstrumentClick={onInstrumentSelect || (() => {})}
                    compact={compactMode}
                  />
                )}
                size={compactMode ? 'small' : 'default'}
              />
            </SortableContext>
          </DndContext>
        )}
      </Card>

      {/* Create Watchlist Modal */}
      <Modal
        title="Create New Watchlist"
        open={createModalVisible}
        onOk={handleCreateWatchlist}
        onCancel={() => {
          setCreateModalVisible(false)
          createForm.resetFields()
        }}
        okText="Create"
      >
        <Form form={createForm} layout="vertical">
          <Form.Item
            name="name"
            label="Watchlist Name"
            rules={[{ required: true, message: 'Please enter a watchlist name' }]}
          >
            <Input placeholder="e.g., Tech Stocks, Forex Pairs" />
          </Form.Item>
          <Form.Item name="description" label="Description (Optional)">
            <Input.TextArea
              rows={2}
              placeholder="Optional description for this watchlist..."
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* Import/Export Modal */}
      <WatchlistImportExport
        visible={importExportVisible}
        onClose={() => setImportExportVisible(false)}
        mode={importExportMode}
        watchlistId={selectedWatchlistId}
      />
    </div>
  )
}