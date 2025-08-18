# API Documentation - Authentication Endpoints

## Overview
This document provides detailed API documentation for the authentication endpoints implemented in Story 1.4: Authentication & Session Management.

## Base URL
- **Development**: `http://localhost:8002`
- **API Prefix**: `/api/v1/auth`

## Authentication Methods

The API supports two authentication methods:
1. **Username/Password**: Traditional credentials-based authentication
2. **API Key**: Token-based authentication for programmatic access

## Endpoints

### 1. User Login
**Endpoint**: `POST /api/v1/auth/login`  
**Description**: Authenticate user and receive JWT tokens

#### Request Headers
```
Content-Type: application/json
```

#### Request Body Options

##### Username/Password Authentication
```json
{
  "username": "admin",
  "password": "admin123"
}
```

##### API Key Authentication
```json
{
  "api_key": "your_api_key_here"
}
```

#### Response (200 OK)
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

#### Error Responses

##### 401 Unauthorized - Invalid Credentials
```json
{
  "detail": "Invalid username or password"
}
```

##### 401 Unauthorized - Invalid API Key
```json
{
  "detail": "Invalid API key"
}
```

##### 422 Unprocessable Entity - Missing Credentials
```json
{
  "detail": "Either username/password or api_key must be provided"
}
```

### 2. Token Refresh
**Endpoint**: `POST /api/v1/auth/refresh`  
**Description**: Refresh JWT access token using refresh token

#### Request Headers
```
Content-Type: application/json
```

#### Request Body (Optional)
```json
{
  "refresh_token": "your_refresh_token_here"
}
```

**Note**: If not provided in body, the refresh token will be read from httpOnly cookie.

#### Response (200 OK)
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

#### Error Responses

##### 401 Unauthorized - Invalid/Missing Refresh Token
```json
{
  "detail": "Refresh token not provided"
}
```

##### 401 Unauthorized - Expired Refresh Token
```json
{
  "detail": "Token has expired"
}
```

### 3. User Logout
**Endpoint**: `POST /api/v1/auth/logout`  
**Description**: Logout user and invalidate tokens

#### Request Headers
```
Authorization: Bearer your_access_token_here
```

#### Response (200 OK)
```json
{
  "message": "Successfully logged out"
}
```

#### Error Responses

##### 401 Unauthorized - Invalid/Missing Token
```json
{
  "detail": "Could not validate credentials"
}
```

### 4. Get Current User
**Endpoint**: `GET /api/v1/auth/me`  
**Description**: Get current authenticated user information

#### Request Headers
```
Authorization: Bearer your_access_token_here
```

#### Response (200 OK)
```json
{
  "id": 1,
  "username": "admin",
  "is_active": true,
  "created_at": "2025-08-16T21:51:23.131081Z",
  "last_login": "2025-08-16T21:52:15.123456Z"
}
```

#### Error Responses

##### 401 Unauthorized - Invalid/Missing Token
```json
{
  "detail": "Could not validate credentials"
}
```

### 5. Validate Token
**Endpoint**: `GET /api/v1/auth/validate`  
**Description**: Validate current authentication token

#### Request Headers
```
Authorization: Bearer your_access_token_here
```

#### Response (200 OK)
```json
{
  "valid": true,
  "user": {
    "id": 1,
    "username": "admin",
    "is_active": true,
    "created_at": "2025-08-16T21:51:23.131081Z",
    "last_login": "2025-08-16T21:52:15.123456Z"
  },
  "message": "Token is valid"
}
```

#### Error Responses

##### 401 Unauthorized - Invalid/Missing Token
```json
{
  "detail": "Could not validate credentials"
}
```

## JWT Token Details

### Token Structure
All JWT tokens contain the following claims:
- `sub`: User ID (subject)
- `exp`: Expiration timestamp
- `iat`: Issued at timestamp
- `jti`: JWT ID (for revocation)
- `type`: Token type ("access" or "refresh")

### Token Lifetimes
- **Access Token**: 24 hours (86400 seconds)
- **Refresh Token**: 7 days (604800 seconds)

### Token Usage
Include access tokens in the Authorization header:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Security Features

### Password Hashing
- Uses bcrypt with salt for secure password storage
- Passwords are never stored in plain text

### Token Security
- JWT tokens signed with HMAC SHA-256
- Refresh tokens stored in httpOnly cookies
- Token revocation on logout
- Automatic token refresh before expiration

### CORS Configuration
- Configured for development environment
- Allows credentials for cookie handling

## Rate Limiting
Currently not implemented but recommended for production:
- Login attempts: 5 per minute per IP
- Token refresh: 10 per minute per user
- General API: 100 per minute per user

## Error Handling

### Standard Error Response Format
```json
{
  "detail": "Error description"
}
```

### HTTP Status Codes
- `200`: Success
- `401`: Unauthorized (invalid credentials/token)
- `422`: Unprocessable Entity (validation error)
- `500`: Internal Server Error

## Testing Examples

### cURL Examples

#### Login with Username/Password
```bash
curl -X POST http://localhost:8002/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

#### Login with API Key
```bash
curl -X POST http://localhost:8002/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your_api_key_here"
  }'
```

#### Access Protected Endpoint
```bash
curl -H "Authorization: Bearer your_access_token_here" \
  http://localhost:8002/api/v1/auth/me
```

#### Refresh Token
```bash
curl -X POST http://localhost:8002/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "your_refresh_token_here"
  }'
```

#### Logout
```bash
curl -X POST http://localhost:8002/api/v1/auth/logout \
  -H "Authorization: Bearer your_access_token_here"
```

### JavaScript/TypeScript Examples

#### Login Function
```typescript
async function login(username: string, password: string) {
  const response = await fetch('http://localhost:8002/api/v1/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ username, password }),
  });
  
  if (!response.ok) {
    throw new Error('Login failed');
  }
  
  return await response.json();
}
```

#### Authenticated Request
```typescript
async function makeAuthenticatedRequest(token: string) {
  const response = await fetch('http://localhost:8002/api/v1/auth/me', {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });
  
  if (!response.ok) {
    throw new Error('Request failed');
  }
  
  return await response.json();
}
```

## Interactive Documentation
For interactive API testing, visit:
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

---

**Version**: 1.0  
**Last Updated**: August 16, 2025  
**Story**: 1.4 Authentication & Session Management