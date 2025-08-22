/**
 * engineService - API service for NautilusTrader engine management
 * 
 * Provides methods for interacting with the Nautilus engine REST API
 * endpoints as defined in Story 6.1.
 */

import axios, { AxiosResponse } from 'axios';

const API_BASE_URL = `${import.meta.env.VITE_API_BASE_URL}/api/v1/nautilus/engine`;

// Types
export interface EngineConfig {
  engine_type: string;
  log_level: string;
  instance_id: string;
  trading_mode: 'paper' | 'live';
  max_memory: string;
  max_cpu: string;
  data_catalog_path: string;
  cache_database_path: string;
  risk_engine_enabled: boolean;
  max_position_size?: number;
  max_order_rate?: number;
}

export interface EngineResponse {
  success: boolean;
  message: string;
  state: string;
  data?: any;
}

export interface EngineStatusResponse {
  success: boolean;
  status: any;
}

export interface BacktestConfig {
  strategy_class: string;
  strategy_config: Record<string, any>;
  start_date: string;
  end_date: string;
  instruments: string[];
  venues?: string[];
  initial_balance?: number;
  base_currency?: string;
  data_sources?: string[];
  bar_types?: string[];
  tick_data?: boolean;
  output_path?: string;
  save_results?: boolean;
}

export interface BacktestResponse {
  success: boolean;
  message?: string;
  backtest_id?: string;
  status?: string;
  backtest?: any;
}

// Configure axios defaults
const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for authentication
axiosInstance.interceptors.request.use(
  (config) => {
    // Add JWT token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('Engine API Error:', error);
    
    if (error.response?.status === 401) {
      // Handle authentication error
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    
    return Promise.reject(error);
  }
);

class EngineService {
  /**
   * Start the NautilusTrader engine
   */
  async start(config: EngineConfig, confirmLiveTrading: boolean = false): Promise<EngineResponse> {
    try {
      const response: AxiosResponse<EngineResponse> = await axiosInstance.post('/start', {
        config,
        confirm_live_trading: confirmLiveTrading
      });
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        message: error.response?.data?.detail || error.message || 'Failed to start engine',
        state: 'error'
      };
    }
  }

  /**
   * Stop the NautilusTrader engine
   */
  async stop(force: boolean = false): Promise<EngineResponse> {
    try {
      const response: AxiosResponse<EngineResponse> = await axiosInstance.post('/stop', {
        force
      });
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        message: error.response?.data?.detail || error.message || 'Failed to stop engine',
        state: 'error'
      };
    }
  }

  /**
   * Restart the NautilusTrader engine
   */
  async restart(): Promise<EngineResponse> {
    try {
      const response: AxiosResponse<EngineResponse> = await axiosInstance.post('/restart');
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        message: error.response?.data?.detail || error.message || 'Failed to restart engine',
        state: 'error'
      };
    }
  }

  /**
   * Get current engine status
   */
  async getStatus(): Promise<EngineStatusResponse> {
    try {
      const response: AxiosResponse<EngineStatusResponse> = await axiosInstance.get('/status');
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        status: {
          error: error.response?.data?.detail || error.message || 'Failed to get engine status'
        }
      };
    }
  }

  /**
   * Update engine configuration
   */
  async updateConfig(config: EngineConfig): Promise<EngineResponse> {
    try {
      const response: AxiosResponse<EngineResponse> = await axiosInstance.put('/config', {
        config
      });
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        message: error.response?.data?.detail || error.message || 'Failed to update configuration',
        state: 'error'
      };
    }
  }

  /**
   * Get engine logs
   */
  async getLogs(lines: number = 100): Promise<{ success: boolean; logs?: string[]; error?: string }> {
    try {
      const response = await axiosInstance.get(`/logs?lines=${lines}`);
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message || 'Failed to get engine logs'
      };
    }
  }

  /**
   * Emergency stop - immediate force stop
   */
  async emergencyStop(): Promise<EngineResponse> {
    try {
      const response: AxiosResponse<EngineResponse> = await axiosInstance.post('/emergency-stop');
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        message: error.response?.data?.detail || error.message || 'Emergency stop failed',
        state: 'error'
      };
    }
  }

  /**
   * Get engine health check
   */
  async healthCheck(): Promise<{ service: string; status: string; timestamp: string; engine_state?: string; error?: string }> {
    try {
      const response = await axiosInstance.get('/health');
      
      return response.data;
    } catch (error: any) {
      return {
        service: 'nautilus-engine-api',
        status: 'error',
        timestamp: new Date().toISOString(),
        error: error.response?.data?.detail || error.message || 'Health check failed'
      };
    }
  }

  /**
   * Start a backtest
   */
  async startBacktest(backtestId: string, config: BacktestConfig): Promise<BacktestResponse> {
    try {
      const response: AxiosResponse<BacktestResponse> = await axiosInstance.post('/backtest', {
        backtest_id: backtestId,
        config
      });
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        message: error.response?.data?.detail || error.message || 'Failed to start backtest'
      };
    }
  }

  /**
   * Get backtest status
   */
  async getBacktestStatus(backtestId: string): Promise<BacktestResponse> {
    try {
      const response: AxiosResponse<BacktestResponse> = await axiosInstance.get(`/backtest/${backtestId}`);
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        message: error.response?.data?.detail || error.message || 'Failed to get backtest status'
      };
    }
  }

  /**
   * Cancel a backtest
   */
  async cancelBacktest(backtestId: string): Promise<BacktestResponse> {
    try {
      const response: AxiosResponse<BacktestResponse> = await axiosInstance.delete(`/backtest/${backtestId}`);
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        message: error.response?.data?.detail || error.message || 'Failed to cancel backtest'
      };
    }
  }

  /**
   * List all backtests
   */
  async listBacktests(): Promise<{ success: boolean; backtests?: any[]; total_count?: number; error?: string }> {
    try {
      const response = await axiosInstance.get('/backtests');
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message || 'Failed to list backtests'
      };
    }
  }

  /**
   * Get data catalog information
   */
  async getDataCatalog(): Promise<{ success: boolean; catalog?: any; error?: string }> {
    try {
      const response = await axiosInstance.get('/catalog');
      
      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message || 'Failed to get data catalog'
      };
    }
  }

  /**
   * Utility method to check if engine is running
   */
  async isEngineRunning(): Promise<boolean> {
    try {
      const status = await this.getStatus();
      return status.success && status.status.state === 'running';
    } catch (error) {
      return false;
    }
  }

  /**
   * Utility method to wait for engine state change
   */
  async waitForState(targetState: string, maxWaitTime: number = 30000, pollInterval: number = 1000): Promise<boolean> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWaitTime) {
      try {
        const status = await this.getStatus();
        if (status.success && status.status.state === targetState) {
          return true;
        }
        
        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, pollInterval));
      } catch (error) {
        console.error('Error polling engine state:', error);
      }
    }
    
    return false;
  }
}

// Create and export singleton instance
export const engineService = new EngineService();
export default engineService;