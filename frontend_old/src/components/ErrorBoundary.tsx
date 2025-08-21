import { Component, ErrorInfo, ReactNode } from 'react';
import { Alert, Button, Card } from 'antd';
import { ExclamationCircleOutlined, ReloadOutlined } from '@ant-design/icons';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
  fallbackMessage?: string;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo
    });
  }

  private handleReload = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  public render() {
    if (this.state.hasError) {
      const { fallbackTitle = 'Component Error', fallbackMessage = 'This component encountered an error and cannot be displayed.' } = this.props;
      
      return (
        <Card>
          <Alert
            message={fallbackTitle}
            description={
              <div>
                <p>{fallbackMessage}</p>
                {this.state.error && (
                  <details style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
                    <summary>Error Details</summary>
                    <pre style={{ whiteSpace: 'pre-wrap', marginTop: 8 }}>
                      {this.state.error.toString()}
                      {this.state.errorInfo?.componentStack}
                    </pre>
                  </details>
                )}
              </div>
            }
            type="error"
            icon={<ExclamationCircleOutlined />}
            action={
              <Button 
                size="small" 
                icon={<ReloadOutlined />} 
                onClick={this.handleReload}
              >
                Retry
              </Button>
            }
          />
        </Card>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;