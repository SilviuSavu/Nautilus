/**
 * Authentication service for API communication
 */

import axios, { AxiosResponse } from 'axios';
import { LoginRequest, TokenResponse, User, ValidationResponse } from '../types/auth';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002';

// Create axios instance for auth requests
const authApi = axios.create({
  baseURL: `${API_BASE_URL}/api/v1/auth`,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Include cookies for refresh token
});

// Token storage keys
const ACCESS_TOKEN_KEY = 'nautilus_access_token';
const TOKEN_EXPIRY_KEY = 'nautilus_token_expiry';

export class AuthService {
  /**
   * Store access token in localStorage
   */
  static setAccessToken(token: string, expiresIn: number): void {
    localStorage.setItem(ACCESS_TOKEN_KEY, token);
    const expiryTime = Date.now() + (expiresIn * 1000);
    localStorage.setItem(TOKEN_EXPIRY_KEY, expiryTime.toString());
  }

  /**
   * Get access token from localStorage
   */
  static getAccessToken(): string | null {
    const token = localStorage.getItem(ACCESS_TOKEN_KEY);
    const expiry = localStorage.getItem(TOKEN_EXPIRY_KEY);
    
    if (!token || !expiry) {
      return null;
    }
    
    // Check if token is expired
    if (Date.now() >= parseInt(expiry)) {
      AuthService.clearTokens();
      return null;
    }
    
    return token;
  }

  /**
   * Clear all stored tokens
   */
  static clearTokens(): void {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(TOKEN_EXPIRY_KEY);
  }

  /**
   * Check if token is expired or will expire soon (within 5 minutes)
   */
  static isTokenExpiringSoon(): boolean {
    const expiry = localStorage.getItem(TOKEN_EXPIRY_KEY);
    if (!expiry) return true;
    
    const expiryTime = parseInt(expiry);
    const fiveMinutesFromNow = Date.now() + (5 * 60 * 1000);
    
    return expiryTime <= fiveMinutesFromNow;
  }

  /**
   * Login with username/password or API key
   */
  static async login(credentials: LoginRequest): Promise<{ user: User; tokens: TokenResponse }> {
    try {
      const response: AxiosResponse<TokenResponse> = await authApi.post('/login', credentials);
      const tokens = response.data;
      
      // Store access token
      AuthService.setAccessToken(tokens.access_token, tokens.expires_in);
      
      // Get user info
      const user = await AuthService.getCurrentUser(tokens.access_token);
      
      return { user, tokens };
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'Login failed');
      }
      throw error;
    }
  }

  /**
   * Logout user
   */
  static async logout(): Promise<void> {
    const token = AuthService.getAccessToken();
    
    try {
      if (token) {
        await authApi.post('/logout', {}, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
      }
    } catch (error) {
      // Continue with logout even if API call fails
      console.warn('Logout API call failed:', error);
    } finally {
      // Always clear local tokens
      AuthService.clearTokens();
    }
  }

  /**
   * Refresh access token
   */
  static async refreshToken(): Promise<TokenResponse> {
    try {
      const response: AxiosResponse<TokenResponse> = await authApi.post('/refresh');
      const tokens = response.data;
      
      // Store new access token
      AuthService.setAccessToken(tokens.access_token, tokens.expires_in);
      
      return tokens;
    } catch (error) {
      // Clear tokens if refresh fails
      AuthService.clearTokens();
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'Token refresh failed');
      }
      throw error;
    }
  }

  /**
   * Get current user info
   */
  static async getCurrentUser(token?: string): Promise<User> {
    const accessToken = token || AuthService.getAccessToken();
    
    if (!accessToken) {
      throw new Error('No access token available');
    }
    
    try {
      const response: AxiosResponse<User> = await authApi.get('/me', {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });
      
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'Failed to get user info');
      }
      throw error;
    }
  }

  /**
   * Validate current token
   */
  static async validateToken(): Promise<ValidationResponse> {
    const token = AuthService.getAccessToken();
    
    if (!token) {
      throw new Error('No access token available');
    }
    
    try {
      const response: AxiosResponse<ValidationResponse> = await authApi.get('/validate', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      
      return response.data;
    } catch (error) {
      // Clear tokens if validation fails
      AuthService.clearTokens();
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'Token validation failed');
      }
      throw error;
    }
  }

  /**
   * Setup axios interceptor for automatic token refresh
   */
  static setupInterceptors(): void {
    // Request interceptor to add auth header
    authApi.interceptors.request.use(
      (config) => {
        const token = AuthService.getAccessToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for automatic token refresh
    authApi.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            await AuthService.refreshToken();
            const token = AuthService.getAccessToken();
            if (token) {
              originalRequest.headers.Authorization = `Bearer ${token}`;
              return authApi(originalRequest);
            }
          } catch (refreshError) {
            // Refresh failed, redirect to login
            AuthService.clearTokens();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );
  }
}

// Setup interceptors when the service is imported
AuthService.setupInterceptors();