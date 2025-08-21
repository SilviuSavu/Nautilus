/**
 * Header component with authentication status and logout
 */

import { Layout, Dropdown, Button, Avatar, Space, Typography } from 'antd';
import { UserOutlined, LogoutOutlined, DownOutlined } from '@ant-design/icons';
import { useAuth } from '../hooks/useAuth';
import dayjs from 'dayjs';

const { Header } = Layout;
const { Text } = Typography;

export function AuthHeader() {
  const { user, logout, isAuthenticated } = useAuth();

  const handleLogout = async () => {
    await logout();
  };

  if (!isAuthenticated || !user) {
    return null;
  }

  const userMenu = {
    items: [
      {
        key: 'user-info',
        label: (
          <div style={{ padding: '8px 0' }}>
            <div><strong>{user.username}</strong></div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              ID: {user.id}
            </Text>
            {user.last_login && (
              <div>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  Last login: {dayjs(user.last_login).format('MMM DD, HH:mm')}
                </Text>
              </div>
            )}
          </div>
        ),
        disabled: true,
      },
      {
        type: 'divider' as const,
      },
      {
        key: 'logout',
        label: (
          <Space>
            <LogoutOutlined />
            Logout
          </Space>
        ),
        onClick: handleLogout,
      },
    ],
  };

  return (
    <Header 
      style={{ 
        background: '#fff', 
        padding: '0 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '1px solid #f0f0f0'
      }}
    >
      <div>
        <Typography.Title level={4} style={{ margin: 0 }}>
          Nautilus Trader Dashboard
        </Typography.Title>
      </div>

      <Dropdown menu={userMenu} trigger={['click']}>
        <Button type="text">
          <Space>
            <Avatar icon={<UserOutlined />} size="small" />
            <span>{user.username}</span>
            <DownOutlined />
          </Space>
        </Button>
      </Dropdown>
    </Header>
  );
}