/**
 * Responsive Layout System
 * Production-ready responsive layout with mobile-first design
 */

import React, { memo, useCallback, useEffect, useState } from 'react';
import { Layout, Grid, Drawer, Button, Space, Dropdown, Switch, Tooltip } from 'antd';
import {
  MenuOutlined,
  CloseOutlined,
  FullscreenOutlined,
  CompressOutlined,
  SettingOutlined,
  MoonOutlined,
  SunOutlined,
  ExpandOutlined,
  ShrinkOutlined
} from '@ant-design/icons';
import { BREAKPOINTS } from '../../styles/theme';
import { ACCESSIBILITY } from '../../constants/ui';

const { Header, Sider, Content, Footer } = Layout;
const { useBreakpoint } = Grid;

export interface ResponsiveLayoutProps {
  /** Layout children */
  children: React.ReactNode;
  /** Header content */
  header?: React.ReactNode;
  /** Sidebar content */
  sidebar?: React.ReactNode;
  /** Footer content */
  footer?: React.ReactNode;
  /** Layout title */
  title?: string;
  /** Logo component */
  logo?: React.ReactNode;
  /** Navigation menu items */
  menuItems?: React.ReactNode;
  /** Header actions */
  headerActions?: React.ReactNode;
  /** Enable sidebar */
  enableSidebar?: boolean;
  /** Enable responsive behavior */
  enableResponsive?: boolean;
  /** Enable fullscreen mode */
  enableFullscreen?: boolean;
  /** Enable dark mode toggle */
  enableDarkMode?: boolean;
  /** Initial dark mode state */
  initialDarkMode?: boolean;
  /** Collapsed sidebar by default */
  defaultCollapsed?: boolean;
  /** Sidebar width */
  sidebarWidth?: number;
  /** Collapsed sidebar width */
  collapsedWidth?: number;
  /** Header height */
  headerHeight?: number;
  /** Footer height */
  footerHeight?: number;
  /** Enable sticky header */
  stickyHeader?: boolean;
  /** Enable sticky footer */
  stickyFooter?: boolean;
  /** Layout className */
  className?: string;
  /** Layout style */
  style?: React.CSSProperties;
  /** Sidebar collapse callback */
  onSidebarToggle?: (collapsed: boolean) => void;
  /** Dark mode change callback */
  onDarkModeChange?: (darkMode: boolean) => void;
  /** Fullscreen change callback */
  onFullscreenChange?: (fullscreen: boolean) => void;
}

const ResponsiveLayout: React.FC<ResponsiveLayoutProps> = memo(({
  children,
  header,
  sidebar,
  footer,
  title,
  logo,
  menuItems,
  headerActions,
  enableSidebar = true,
  enableResponsive = true,
  enableFullscreen = true,
  enableDarkMode = true,
  initialDarkMode = false,
  defaultCollapsed = false,
  sidebarWidth = 280,
  collapsedWidth = 80,
  headerHeight = 64,
  footerHeight = 48,
  stickyHeader = true,
  stickyFooter = false,
  className,
  style,
  onSidebarToggle,
  onDarkModeChange,
  onFullscreenChange
}) => {
  // State management
  const [sidebarCollapsed, setSidebarCollapsed] = useState(defaultCollapsed);
  const [mobileDrawerVisible, setMobileDrawerVisible] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(initialDarkMode);

  // Responsive breakpoints
  const screens = useBreakpoint();
  const isMobile = !screens.md;
  const isTablet = screens.md && !screens.lg;
  const isDesktop = screens.lg;

  // Auto-collapse sidebar on mobile/tablet
  useEffect(() => {
    if (enableResponsive) {
      if (isMobile) {
        setSidebarCollapsed(true);
      } else if (isTablet) {
        setSidebarCollapsed(true);
      } else {
        setSidebarCollapsed(defaultCollapsed);
      }
    }
  }, [isMobile, isTablet, enableResponsive, defaultCollapsed]);

  // Handle sidebar toggle
  const handleSidebarToggle = useCallback(() => {
    if (isMobile) {
      setMobileDrawerVisible(!mobileDrawerVisible);
    } else {
      const newCollapsed = !sidebarCollapsed;
      setSidebarCollapsed(newCollapsed);
      onSidebarToggle?.(newCollapsed);
    }
  }, [isMobile, mobileDrawerVisible, sidebarCollapsed, onSidebarToggle]);

  // Handle fullscreen toggle
  const handleFullscreenToggle = useCallback(async () => {
    try {
      if (!isFullscreen) {
        await document.documentElement.requestFullscreen();
        setIsFullscreen(true);
      } else {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
      onFullscreenChange?.(isFullscreen);
    } catch (error) {
      console.error('Fullscreen toggle failed:', error);
    }
  }, [isFullscreen, onFullscreenChange]);

  // Handle dark mode toggle
  const handleDarkModeToggle = useCallback(() => {
    const newDarkMode = !isDarkMode;
    setIsDarkMode(newDarkMode);
    onDarkModeChange?.(newDarkMode);
    
    // Apply theme class to body
    document.body.classList.toggle('dark', newDarkMode);
  }, [isDarkMode, onDarkModeChange]);

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      const isCurrentlyFullscreen = Boolean(document.fullscreenElement);
      setIsFullscreen(isCurrentlyFullscreen);
      onFullscreenChange?.(isCurrentlyFullscreen);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, [onFullscreenChange]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      // Only handle shortcuts when no input is focused
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (event.key.toLowerCase()) {
        case ACCESSIBILITY.SHORTCUTS.FULLSCREEN:
          if (enableFullscreen) {
            event.preventDefault();
            handleFullscreenToggle();
          }
          break;
        case ACCESSIBILITY.SHORTCUTS.SETTINGS:
          if (enableSidebar) {
            event.preventDefault();
            handleSidebarToggle();
          }
          break;
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [enableFullscreen, enableSidebar, handleFullscreenToggle, handleSidebarToggle]);

  // Calculate responsive sidebar width
  const getResponsiveSidebarWidth = () => {
    if (isMobile) return 280;
    if (sidebarCollapsed) return collapsedWidth;
    return sidebarWidth;
  };

  // Header component
  const renderHeader = () => (
    <Header
      style={{
        height: headerHeight,
        lineHeight: `${headerHeight}px`,
        padding: isMobile ? '0 16px' : '0 24px',
        background: isDarkMode ? '#141414' : '#fff',
        borderBottom: `1px solid ${isDarkMode ? '#303030' : '#f0f0f0'}`,
        position: stickyHeader ? 'sticky' : 'static',
        top: 0,
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
        {enableSidebar && (
          <Button
            type="text"
            icon={<MenuOutlined />}
            onClick={handleSidebarToggle}
            style={{ marginRight: 16 }}
            aria-label={ACCESSIBILITY.LABELS.SETTINGS_BUTTON}
          />
        )}
        
        {logo && (
          <div style={{ marginRight: 24 }}>
            {logo}
          </div>
        )}
        
        {title && !isMobile && (
          <h1 style={{ margin: 0, fontSize: 20, fontWeight: 600 }}>
            {title}
          </h1>
        )}
        
        {menuItems && !isMobile && (
          <div style={{ marginLeft: 'auto', marginRight: 24 }}>
            {menuItems}
          </div>
        )}
      </div>
      
      <Space size={isMobile ? 8 : 16}>
        {headerActions}
        
        {enableDarkMode && (
          <Tooltip title="Toggle dark mode">
            <Button
              type="text"
              icon={isDarkMode ? <SunOutlined /> : <MoonOutlined />}
              onClick={handleDarkModeToggle}
              size={isMobile ? 'small' : 'middle'}
            />
          </Tooltip>
        )}
        
        {enableFullscreen && !isMobile && (
          <Tooltip title={`${isFullscreen ? 'Exit' : 'Enter'} fullscreen (F)`}>
            <Button
              type="text"
              icon={isFullscreen ? <CompressOutlined /> : <FullscreenOutlined />}
              onClick={handleFullscreenToggle}
              aria-label={ACCESSIBILITY.LABELS.FULLSCREEN_BUTTON}
            />
          </Tooltip>
        )}
        
        {header}
      </Space>
    </Header>
  );

  // Sidebar component
  const renderSidebar = () => {
    if (!enableSidebar || !sidebar) return null;

    const sidebarContent = (
      <div
        style={{
          height: '100%',
          background: isDarkMode ? '#141414' : '#fff',
          borderRight: `1px solid ${isDarkMode ? '#303030' : '#f0f0f0'}`,
          overflow: 'auto'
        }}
      >
        {sidebar}
      </div>
    );

    if (isMobile) {
      return (
        <Drawer
          title={title}
          placement="left"
          onClose={() => setMobileDrawerVisible(false)}
          open={mobileDrawerVisible}
          bodyStyle={{ padding: 0 }}
          width={280}
          closeIcon={<CloseOutlined />}
        >
          {sidebarContent}
        </Drawer>
      );
    }

    return (
      <Sider
        width={sidebarWidth}
        collapsedWidth={collapsedWidth}
        collapsed={sidebarCollapsed}
        trigger={null}
        style={{
          height: '100vh',
          position: 'fixed',
          left: 0,
          top: 0,
          zIndex: 999,
          background: isDarkMode ? '#141414' : '#fff'
        }}
        theme={isDarkMode ? 'dark' : 'light'}
        collapsible
      >
        {sidebarContent}
      </Sider>
    );
  };

  // Footer component
  const renderFooter = () => {
    if (!footer) return null;

    return (
      <Footer
        style={{
          height: footerHeight,
          lineHeight: `${footerHeight}px`,
          padding: isMobile ? '0 16px' : '0 24px',
          background: isDarkMode ? '#141414' : '#fff',
          borderTop: `1px solid ${isDarkMode ? '#303030' : '#f0f0f0'}`,
          position: stickyFooter ? 'sticky' : 'static',
          bottom: 0,
          zIndex: 1000,
          textAlign: 'center'
        }}
      >
        {footer}
      </Footer>
    );
  };

  // Calculate content margin for desktop sidebar
  const getContentMarginLeft = () => {
    if (isMobile || !enableSidebar) return 0;
    return getResponsiveSidebarWidth();
  };

  // Layout styles
  const layoutStyle: React.CSSProperties = {
    minHeight: '100vh',
    background: isDarkMode ? '#000' : '#f5f5f5',
    transition: 'all 0.2s ease',
    ...style
  };

  const contentStyle: React.CSSProperties = {
    marginLeft: getContentMarginLeft(),
    transition: 'margin-left 0.2s ease',
    display: 'flex',
    flexDirection: 'column',
    minHeight: '100vh',
    background: 'transparent'
  };

  const mainContentStyle: React.CSSProperties = {
    flex: 1,
    padding: isMobile ? 16 : 24,
    overflow: 'auto',
    background: 'transparent'
  };

  return (
    <div className={className} style={{ position: 'relative' }}>
      <Layout style={layoutStyle}>
        {renderSidebar()}
        
        <Layout style={contentStyle}>
          {renderHeader()}
          
          <Content style={mainContentStyle}>
            {children}
          </Content>
          
          {renderFooter()}
        </Layout>
      </Layout>

      {/* Mobile overlay */}
      {isMobile && mobileDrawerVisible && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.45)',
            zIndex: 998
          }}
          onClick={() => setMobileDrawerVisible(false)}
        />
      )}
    </div>
  );
});

ResponsiveLayout.displayName = 'ResponsiveLayout';

export default ResponsiveLayout;