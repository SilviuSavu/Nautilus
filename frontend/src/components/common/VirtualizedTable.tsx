/**
 * Virtualized Table Component
 * High-performance table with virtualization for large datasets
 */

import React, { memo, useMemo, useCallback, useState, useRef } from 'react';
import { Table, Input, Button, Space, Typography, Card, Tag } from 'antd';
import { FixedSizeList as List } from 'react-window';
import {
  SearchOutlined,
  FilterOutlined,
  DownloadOutlined,
  ReloadOutlined,
  FullscreenOutlined,
  CompressOutlined
} from '@ant-design/icons';
import type { ColumnType } from 'antd/es/table';
import { UI_CONSTANTS } from '../../constants/ui';

const { Text } = Typography;

export interface VirtualizedTableColumn<T = any> extends ColumnType<T> {
  /** Enable virtual scrolling for this column */
  virtualized?: boolean;
  /** Custom cell renderer for better performance */
  cellRenderer?: (value: any, record: T, index: number) => React.ReactNode;
}

export interface VirtualizedTableProps<T = any> {
  /** Table data */
  dataSource: T[];
  /** Table columns */
  columns: VirtualizedTableColumn<T>[];
  /** Table height for virtualization */
  height?: number;
  /** Row height for virtualization */
  rowHeight?: number;
  /** Enable virtual scrolling */
  enableVirtualization?: boolean;
  /** Loading state */
  loading?: boolean;
  /** Row key */
  rowKey?: string | ((record: T) => string);
  /** Enable search */
  enableSearch?: boolean;
  /** Search placeholder */
  searchPlaceholder?: string;
  /** Enable filtering */
  enableFilter?: boolean;
  /** Enable export */
  enableExport?: boolean;
  /** Export filename */
  exportFilename?: string;
  /** Enable fullscreen */
  enableFullscreen?: boolean;
  /** Pagination configuration */
  pagination?: false | {
    pageSize?: number;
    showSizeChanger?: boolean;
    showQuickJumper?: boolean;
  };
  /** Row selection */
  rowSelection?: {
    onChange?: (selectedRowKeys: React.Key[], selectedRows: T[]) => void;
    getCheckboxProps?: (record: T) => { disabled?: boolean; name?: string };
  };
  /** Table title */
  title?: string;
  /** Refresh callback */
  onRefresh?: () => void;
  /** Row click handler */
  onRowClick?: (record: T, index: number) => void;
  /** Custom row className */
  rowClassName?: (record: T, index: number) => string;
  /** Additional table props */
  tableProps?: any;
  /** Container className */
  className?: string;
  /** Container styles */
  style?: React.CSSProperties;
}

const VirtualizedTable = <T extends Record<string, any>>({
  dataSource,
  columns,
  height = 400,
  rowHeight = 54,
  enableVirtualization = false,
  loading = false,
  rowKey = 'id',
  enableSearch = true,
  searchPlaceholder = 'Search...',
  enableFilter = true,
  enableExport = true,
  exportFilename = 'table-data',
  enableFullscreen = true,
  pagination = { pageSize: 50, showSizeChanger: true, showQuickJumper: true },
  rowSelection,
  title,
  onRefresh,
  onRowClick,
  rowClassName,
  tableProps = {},
  className,
  style
}: VirtualizedTableProps<T>): React.ReactElement => {
  const [searchText, setSearchText] = useState('');
  const [filteredData, setFilteredData] = useState<T[]>(dataSource);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);
  const tableRef = useRef<HTMLDivElement>(null);

  // Filter data based on search
  const searchedData = useMemo(() => {
    if (!searchText.trim()) return dataSource;
    
    return dataSource.filter(record =>
      Object.values(record).some(value =>
        String(value).toLowerCase().includes(searchText.toLowerCase())
      )
    );
  }, [dataSource, searchText]);

  // Handle search
  const handleSearch = useCallback((value: string) => {
    setSearchText(value);
  }, []);

  // Handle export
  const handleExport = useCallback(() => {
    const exportData = searchedData.map(record => {
      const exportRecord: any = {};
      columns.forEach(col => {
        if (col.dataIndex && typeof col.dataIndex === 'string') {
          exportRecord[col.title as string] = record[col.dataIndex];
        }
      });
      return exportRecord;
    });

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `${exportFilename}-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [searchedData, columns, exportFilename]);

  // Toggle fullscreen
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => !prev);
  }, []);

  // Optimized columns with custom renderers
  const optimizedColumns = useMemo(() => {
    return columns.map(col => ({
      ...col,
      render: col.cellRenderer || col.render,
      shouldCellUpdate: (record: T, prevRecord: T) => {
        if (!col.dataIndex || typeof col.dataIndex !== 'string') return true;
        return record[col.dataIndex] !== prevRecord[col.dataIndex];
      }
    }));
  }, [columns]);

  // Virtual row renderer
  const VirtualRow = useCallback(({ index, style }: { index: number; style: React.CSSProperties }) => {
    const record = searchedData[index];
    if (!record) return null;

    return (
      <div
        style={style}
        className={rowClassName ? rowClassName(record, index) : undefined}
        onClick={() => onRowClick?.(record, index)}
      >
        <div style={{ display: 'flex', alignItems: 'center', padding: '0 16px', height: rowHeight }}>
          {columns.map((col, colIndex) => {
            const value = col.dataIndex && typeof col.dataIndex === 'string' 
              ? record[col.dataIndex] 
              : undefined;
            
            return (
              <div 
                key={colIndex}
                style={{ 
                  flex: col.width ? `0 0 ${col.width}px` : 1,
                  padding: '0 8px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}
              >
                {col.cellRenderer ? col.cellRenderer(value, record, index) : value}
              </div>
            );
          })}
        </div>
      </div>
    );
  }, [searchedData, columns, rowHeight, rowClassName, onRowClick]);

  // Table configuration
  const tableConfig = {
    dataSource: searchedData,
    columns: optimizedColumns,
    rowKey,
    loading,
    pagination,
    rowSelection: rowSelection ? {
      ...rowSelection,
      selectedRowKeys,
      onChange: (keys: React.Key[], rows: T[]) => {
        setSelectedRowKeys(keys);
        rowSelection.onChange?.(keys, rows);
      }
    } : undefined,
    scroll: enableVirtualization ? { y: height } : { x: 1000 },
    onRow: onRowClick ? (record: T, index: number) => ({
      onClick: () => onRowClick(record, index || 0)
    }) : undefined,
    components: enableVirtualization ? {
      body: {
        wrapper: ({ children }: any) => (
          <List
            height={height}
            itemCount={searchedData.length}
            itemSize={rowHeight}
            width="100%"
          >
            {VirtualRow}
          </List>
        )
      }
    } : undefined,
    ...tableProps
  };

  const containerStyle: React.CSSProperties = {
    position: isFullscreen ? 'fixed' : 'relative',
    top: isFullscreen ? 0 : 'auto',
    left: isFullscreen ? 0 : 'auto',
    width: isFullscreen ? '100vw' : '100%',
    height: isFullscreen ? '100vh' : 'auto',
    zIndex: isFullscreen ? 1000 : 'auto',
    backgroundColor: isFullscreen ? '#fff' : 'transparent',
    padding: isFullscreen ? '16px' : '0',
    ...style
  };

  return (
    <Card
      ref={tableRef}
      style={containerStyle}
      className={className}
      title={title && (
        <Space>
          <Text strong>{title}</Text>
          <Tag color="blue">{searchedData.length} items</Tag>
          {selectedRowKeys.length > 0 && (
            <Tag color="green">{selectedRowKeys.length} selected</Tag>
          )}
        </Space>
      )}
      extra={
        <Space size="small">
          {enableSearch && (
            <Input
              placeholder={searchPlaceholder}
              prefix={<SearchOutlined />}
              value={searchText}
              onChange={(e) => handleSearch(e.target.value)}
              style={{ width: 200 }}
              size="small"
              allowClear
            />
          )}
          {enableFilter && (
            <Button
              size="small"
              icon={<FilterOutlined />}
              onClick={() => {/* Implement filter modal */}}
            >
              Filter
            </Button>
          )}
          {onRefresh && (
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={onRefresh}
              loading={loading}
            >
              Refresh
            </Button>
          )}
          {enableExport && (
            <Button
              size="small"
              icon={<DownloadOutlined />}
              onClick={handleExport}
              disabled={searchedData.length === 0}
            >
              Export
            </Button>
          )}
          {enableFullscreen && (
            <Button
              size="small"
              icon={isFullscreen ? <CompressOutlined /> : <FullscreenOutlined />}
              onClick={toggleFullscreen}
            />
          )}
        </Space>
      }
    >
      <Table {...tableConfig} />
    </Card>
  );
};

export default memo(VirtualizedTable) as <T extends Record<string, any>>(
  props: VirtualizedTableProps<T>
) => React.ReactElement;