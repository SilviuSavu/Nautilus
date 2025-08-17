/**
 * Login page component
 */

import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Card, Form, Input, Button, Tabs, Alert, Typography, Space, Divider } from 'antd';
import { UserOutlined, LockOutlined, KeyOutlined } from '@ant-design/icons';
import { useAuth } from '../hooks/useAuth';
import { LoginRequest } from '../types/auth';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

export function Login() {
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('username');
  const { login, isAuthenticated, error, clearError } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Get the redirect path from navigation state or default to dashboard
  const from = (location.state as any)?.from?.pathname || '/dashboard';

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, from]);

  // Clear error when tab changes
  useEffect(() => {
    clearError();
  }, [activeTab, clearError]);

  const handleLogin = async (values: any) => {
    setLoading(true);
    clearError();

    try {
      const credentials: LoginRequest = activeTab === 'username' 
        ? { username: values.username, password: values.password }
        : { api_key: values.api_key };

      await login(credentials);
      // Navigation will happen automatically via useEffect when isAuthenticated changes
    } catch (error) {
      // Error will be displayed via the error state from useAuth
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px'
    }}>
      <Card 
        style={{ 
          width: '100%', 
          maxWidth: '400px',
          boxShadow: '0 10px 30px rgba(0,0,0,0.3)'
        }}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <div style={{ textAlign: 'center' }}>
            <Title level={2} style={{ marginBottom: '8px' }}>
              Nautilus Trader
            </Title>
            <Text type="secondary">
              Sign in to access your trading dashboard
            </Text>
          </div>

          {error && (
            <Alert
              message="Authentication Failed"
              description={error}
              type="error"
              showIcon
              closable
              onClose={clearError}
            />
          )}

          <Tabs 
            activeKey={activeTab} 
            onChange={setActiveTab}
            centered
          >
            <TabPane 
              tab={
                <span>
                  <UserOutlined />
                  Username & Password
                </span>
              } 
              key="username"
            >
              <Form
                name="username-login"
                onFinish={handleLogin}
                layout="vertical"
                size="large"
              >
                <Form.Item
                  name="username"
                  label="Username"
                  rules={[{ required: true, message: 'Please enter your username' }]}
                >
                  <Input
                    prefix={<UserOutlined />}
                    placeholder="Enter username"
                    autoComplete="username"
                  />
                </Form.Item>

                <Form.Item
                  name="password"
                  label="Password"
                  rules={[{ required: true, message: 'Please enter your password' }]}
                >
                  <Input.Password
                    prefix={<LockOutlined />}
                    placeholder="Enter password"
                    autoComplete="current-password"
                  />
                </Form.Item>

                <Form.Item style={{ marginBottom: 0 }}>
                  <Button
                    type="primary"
                    htmlType="submit"
                    loading={loading}
                    style={{ width: '100%' }}
                  >
                    Sign In
                  </Button>
                </Form.Item>
              </Form>
            </TabPane>

            <TabPane 
              tab={
                <span>
                  <KeyOutlined />
                  API Key
                </span>
              } 
              key="api_key"
            >
              <Form
                name="api-key-login"
                onFinish={handleLogin}
                layout="vertical"
                size="large"
              >
                <Form.Item
                  name="api_key"
                  label="API Key"
                  rules={[{ required: true, message: 'Please enter your API key' }]}
                >
                  <Input.Password
                    prefix={<KeyOutlined />}
                    placeholder="Enter API key"
                    autoComplete="off"
                  />
                </Form.Item>

                <Form.Item style={{ marginBottom: 0 }}>
                  <Button
                    type="primary"
                    htmlType="submit"
                    loading={loading}
                    style={{ width: '100%' }}
                  >
                    Sign In
                  </Button>
                </Form.Item>
              </Form>
            </TabPane>
          </Tabs>

          <Divider style={{ margin: '12px 0' }} />

          <div style={{ textAlign: 'center' }}>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Default credentials: admin / admin123
            </Text>
          </div>
        </Space>
      </Card>
    </div>
  );
}