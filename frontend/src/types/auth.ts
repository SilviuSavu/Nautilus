/**
 * Authentication types and interfaces
 */

export interface User {
  id: number;
  username: string;
  is_active: boolean;
  created_at: string;
  last_login?: string;
}

export interface LoginRequest {
  username?: string;
  password?: string;
  api_key?: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

export interface AuthContextType extends AuthState {
  login: (credentials: LoginRequest) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<void>;
  clearError: () => void;
}

export interface AuthResponse {
  user: User;
  tokens: TokenResponse;
}

export interface ValidationResponse {
  valid: boolean;
  user: User;
  message: string;
}