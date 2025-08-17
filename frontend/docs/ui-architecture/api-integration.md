# API Integration

## Service Template

```typescript
import { ApiClient } from './client';
import { 
  OrderRequest, 
  OrderResponse, 
  OrderStatus, 
  OrderUpdatePayload,
  ApiResponse 
} from '../types/api';

export class OrderService {
  private client: ApiClient;

  constructor(client: ApiClient) {
    this.client = client;
  }

  async placeOrder(orderRequest: OrderRequest): Promise<ApiResponse<OrderResponse>> {
    try {
      const response = await this.client.post<OrderResponse>('/orders', {
        body: orderRequest,
        timeout: 5000,
      });

      return {
        success: true,
        data: response.data,
        timestamp: Date.now(),
      };
    } catch (error) {
      return this.handleOrderError(error, 'PLACE_ORDER_FAILED');
    }
  }

  async cancelOrder(orderId: string): Promise<ApiResponse<void>> {
    try {
      await this.client.delete(`/orders/${orderId}`, {
        timeout: 3000,
      });

      return {
        success: true,
        data: undefined,
        timestamp: Date.now(),
      };
    } catch (error) {
      return this.handleOrderError(error, 'CANCEL_ORDER_FAILED');
    }
  }

  private handleOrderError(error: unknown, operation: string): ApiResponse<never> {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    
    console.error(`${operation}:`, error);
    
    return {
      success: false,
      error: {
        code: operation,
        message: errorMessage,
        timestamp: Date.now(),
      },
      timestamp: Date.now(),
    };
  }
}
```

## API Client Configuration

```typescript
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { useSystemStore } from '../store/systemStore';

export interface ApiClientConfig {
  baseURL: string;
  timeout: number;
  apiKey?: string;
  retryAttempts: number;
  retryDelay: number;
}

export class ApiClient {
  private instance: AxiosInstance;
  private config: ApiClientConfig;

  constructor(config: ApiClientConfig) {
    this.config = config;
    this.instance = this.createAxiosInstance();
    this.setupInterceptors();
  }

  private createAxiosInstance(): AxiosInstance {
    return axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });
  }

  private setupInterceptors(): void {
    // Request interceptor for authentication
    this.instance.interceptors.request.use(
      (config) => {
        const { apiKey } = this.config;
        if (apiKey) {
          config.headers.Authorization = `Bearer ${apiKey}`;
        }

        config.metadata = { startTime: Date.now() };
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling and latency tracking
    this.instance.interceptors.response.use(
      (response: AxiosResponse) => {
        const latency = Date.now() - response.config.metadata?.startTime;
        useSystemStore.getState().updateApiLatency(latency);
        return response;
      },
      async (error) => {
        // Handle authentication errors and retries
        if (error.response?.status === 401) {
          useSystemStore.getState().setAuthenticationError();
          return Promise.reject(error);
        }

        if (this.shouldRetry(error)) {
          return this.retryRequest(error.config);
        }

        return Promise.reject(error);
      }
    );
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.get<T>(url, config);
  }

  async post<T>(url: string, config?: AxiosRequestConfig & { body?: any }): Promise<AxiosResponse<T>> {
    const { body, ...axiosConfig } = config || {};
    return this.instance.post<T>(url, body, axiosConfig);
  }
}
```
