/**
 * Enhanced Error Boundary Component
 * Production-ready error handling with detailed reporting and recovery options
 */

import React, { Component, ReactNode, ErrorInfo } from 'react';
import { Result, Button, Card, Typography, Space, Alert, Collapse } from 'antd';
import {
  ExclamationCircleOutlined,
  ReloadOutlined,
  BugOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;

interface ErrorBoundaryProps {
  children: ReactNode;
  fallbackTitle?: string;
  fallbackMessage?: string;
  showErrorDetails?: boolean;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  className?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: ''
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    const errorId = `error-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    return {
      hasError: true,
      error,
      errorId
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      errorInfo
    });

    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.group('🚨 Error Boundary Caught Error');
      console.error('Error:', error);
      console.error('Error Info:', errorInfo);
      console.groupEnd();
    }

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // In production, you might want to send this to an error reporting service
    if (process.env.NODE_ENV === 'production') {
      // Example: sendToErrorReportingService(error, errorInfo);
    }
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: ''
    });
  };

  handleReload = () => {
    window.location.reload();
  };

  copyErrorToClipboard = async () => {
    const { error, errorInfo, errorId } = this.state;
    const errorReport = {
      errorId,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      error: {
        name: error?.name,
        message: error?.message,
        stack: error?.stack
      },
      componentStack: errorInfo?.componentStack
    };

    try {
      await navigator.clipboard.writeText(JSON.stringify(errorReport, null, 2));
      // You could show a notification here
    } catch (err) {
      console.error('Failed to copy error report:', err);
    }
  };

  render() {
    const { 
      children, 
      fallbackTitle = "Something went wrong", 
      fallbackMessage = "An unexpected error has occurred. Please try refreshing the page.",
      showErrorDetails = process.env.NODE_ENV === 'development',
      className 
    } = this.props;

    const { hasError, error, errorInfo, errorId } = this.state;

    if (!hasError) {
      return children;
    }

    return (
      <div className={className} style={{ padding: 24, minHeight: 400 }}>
        <Card>
          <Result
            status="error"
            icon={<ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />}
            title={fallbackTitle}
            subTitle={
              <Space direction="vertical" size="small">
                <Text type="secondary">{fallbackMessage}</Text>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  Error ID: {errorId}
                </Text>
              </Space>
            }
            extra={
              <Space wrap>
                <Button 
                  type="primary" 
                  icon={<ReloadOutlined />}
                  onClick={this.handleRetry}
                >
                  Try Again
                </Button>
                <Button 
                  icon={<ReloadOutlined />}
                  onClick={this.handleReload}
                >
                  Reload Page
                </Button>
                {showErrorDetails && (
                  <Button 
                    icon={<BugOutlined />}
                    onClick={this.copyErrorToClipboard}
                  >
                    Copy Error Report
                  </Button>
                )}
              </Space>
            }
          />

          {showErrorDetails && error && (
            <div style={{ marginTop: 24 }}>
              <Alert
                message="Development Error Details"
                description="This information is only shown in development mode."
                type="info"
                showIcon
                icon={<InfoCircleOutlined />}
                style={{ marginBottom: 16 }}
              />
              
              <Collapse ghost>
                <Panel 
                  header={
                    <Space>
                      <BugOutlined />
                      <Text strong>Error Details</Text>
                    </Space>
                  } 
                  key="error-details"
                >
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Title level={5}>Error Message</Title>
                      <Card size="small">
                        <Text code copyable>
                          {error.message}
                        </Text>
                      </Card>
                    </div>

                    {error.stack && (
                      <div>
                        <Title level={5}>Stack Trace</Title>
                        <Card size="small">
                          <pre style={{ 
                            fontSize: 12, 
                            margin: 0, 
                            whiteSpace: 'pre-wrap',
                            maxHeight: 300,
                            overflow: 'auto'
                          }}>
                            {error.stack}
                          </pre>
                        </Card>
                      </div>
                    )}

                    {errorInfo?.componentStack && (
                      <div>
                        <Title level={5}>Component Stack</Title>
                        <Card size="small">
                          <pre style={{ 
                            fontSize: 12, 
                            margin: 0, 
                            whiteSpace: 'pre-wrap',
                            maxHeight: 200,
                            overflow: 'auto'
                          }}>
                            {errorInfo.componentStack}
                          </pre>
                        </Card>
                      </div>
                    )}
                  </Space>
                </Panel>
              </Collapse>
            </div>
          )}

          <div style={{ marginTop: 24 }}>
            <Paragraph type="secondary" style={{ fontSize: 12 }}>
              If this error persists, please contact the development team with the error ID above.
              In the meantime, you can try refreshing the page or navigating to a different section.
            </Paragraph>
          </div>
        </Card>
      </div>
    );
  }
}

export default ErrorBoundary;