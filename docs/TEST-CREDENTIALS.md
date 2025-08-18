# Test Credentials Reference - Nautilus Trader Authentication

## Overview
This document provides test credentials and access information for QA testing of the Authentication & Session Management system (Story 1.4).

## Default User Account

### Admin User Credentials
- **Username**: `admin`
- **Password**: `admin123`
- **User ID**: 1
- **Account Type**: Administrator
- **Status**: Active

## API Key Authentication

### Getting Current API Key
The API key is dynamically generated when the server starts. To get the current valid API key:

#### Method 1: Debug Endpoint (Development Only)
```bash
curl -s http://localhost:8002/api/v1/auth/debug/admin-api-key | python3 -m json.tool
```

**Sample Response:**
```json
{
    "username": "admin",
    "user_id": 1,
    "api_key": "Yo5uWJVyCTCVwoY2_8kBk49j5PBKdY-ts7ygKUMAFHw",
    "created_at": "2025-08-16T21:51:23.131081+00:00",
    "warning": "This is a debug endpoint - remove in production!"
}
```

#### Method 2: Login and Extract (Production Method)
1. Login with username/password
2. Extract API key from server database (implementation specific)

## Testing Scenarios

### Valid Authentication Tests

#### Username/Password Login
```bash
curl -X POST http://localhost:8002/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

#### API Key Login
```bash
# Replace with current API key from debug endpoint
curl -X POST http://localhost:8002/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "YOUR_CURRENT_API_KEY_HERE"
  }'
```

### Invalid Authentication Tests

#### Wrong Username
```bash
curl -X POST http://localhost:8002/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "wronguser",
    "password": "admin123"
  }'
```

#### Wrong Password
```bash
curl -X POST http://localhost:8002/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "wrongpassword"
  }'
```

#### Invalid API Key
```bash
curl -X POST http://localhost:8002/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "invalid_key_12345"
  }'
```

## Browser Testing

### Access URLs
- **Frontend Login**: http://localhost:3001/login
- **Frontend Dashboard**: http://localhost:3001/dashboard
- **Backend API**: http://localhost:8002
- **API Docs**: http://localhost:8002/docs

### Login Flow Testing
1. Navigate to http://localhost:3001
2. Should auto-redirect to `/login` if not authenticated
3. Use credentials from this document
4. Should redirect to `/dashboard` on successful login

## Token Information

### JWT Token Structure
Successful authentication returns:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Token Expiration
- **Access Token**: 24 hours (86400 seconds)
- **Refresh Token**: 7 days
- **Auto-refresh**: Occurs before access token expiration

### Using Tokens
```bash
# Access protected endpoints with token
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  http://localhost:8002/api/v1/auth/me
```

## Security Testing

### Expected Responses

#### Valid Authentication (200 OK)
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

#### Invalid Credentials (401 Unauthorized)
```json
{
  "detail": "Invalid username or password"
}
```

#### Invalid API Key (401 Unauthorized)
```json
{
  "detail": "Invalid API key"
}
```

#### Missing Credentials (422 Unprocessable Entity)
```json
{
  "detail": "Either username/password or api_key must be provided"
}
```

## Environment Reset

### Clearing Session Data
To reset authentication state:

1. **Browser**: Clear cookies and localStorage
2. **API**: Use logout endpoint to invalidate tokens
3. **Server**: Restart to clear in-memory database

### Server Restart Impact
⚠️ **Important**: When the backend server restarts:
- All API keys are regenerated
- Previous API keys become invalid
- All user sessions are cleared
- You must get new API key from debug endpoint

## Troubleshooting

### Common Issues

#### API Key Not Working
- **Cause**: Server restart regenerated new API key
- **Solution**: Get current API key from debug endpoint

#### Session Not Persisting
- **Cause**: Cookies not set or cleared
- **Solution**: Check browser cookies and ensure httpOnly cookies enabled

#### 401 Errors on Valid Credentials
- **Cause**: Server database state issue
- **Solution**: Restart backend server and retry

#### CORS Errors
- **Cause**: Frontend/backend port mismatch
- **Solution**: Verify ports match between frontend config and backend

### Debug Commands

```bash
# Check server status
curl http://localhost:8002/health

# Check user info with token
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8002/api/v1/auth/me

# Validate token
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8002/api/v1/auth/validate
```

## Production Notes

### Security Considerations
1. **Remove debug endpoint** in production
2. **Use environment variables** for credentials
3. **Implement proper user management** system
4. **Use secure password hashing** (already implemented)
5. **Enable HTTPS** for production

### User Management
Current implementation uses in-memory storage. For production:
- Implement database persistence
- Add user registration/management
- Implement role-based access control
- Add password reset functionality

---

**Last Updated**: August 16, 2025  
**For Story**: 1.4 Authentication & Session Management  
**Environment**: Development/Testing