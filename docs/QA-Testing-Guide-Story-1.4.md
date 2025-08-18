# QA Testing Guide - Story 1.4: Authentication & Session Management

## Overview
This guide provides comprehensive testing instructions for the Authentication and Session Management system (Story 1.4) implemented for the Nautilus Trader platform.

## Test Environment Setup

### Prerequisites
- Node.js and npm installed
- Python 3.9+ installed
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Starting the Application

1. **Start Backend Server:**
   ```bash
   cd /path/to/Nautilus/backend
   python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8002
   ```

2. **Start Frontend Server:**
   ```bash
   cd /path/to/Nautilus/frontend
   npm run dev
   ```

3. **Access URLs:**
   - **Frontend**: http://localhost:3001
   - **Backend API**: http://localhost:8002
   - **API Documentation**: http://localhost:8002/docs

## Test Credentials

### Default User Account
- **Username**: `admin`
- **Password**: `admin123`
- **API Key**: Retrieved via debug endpoint (see API Testing section)

## Testing Scenarios

### 1. Username/Password Authentication

#### Test Case 1.1: Valid Login
**Steps:**
1. Navigate to http://localhost:3001
2. Verify redirect to login page (/login)
3. Ensure "Username & Password" tab is selected
4. Enter username: `admin`
5. Enter password: `admin123`
6. Click "Sign In"

**Expected Results:**
- ✅ Successful authentication
- ✅ Redirect to dashboard (/dashboard)
- ✅ User menu shows "admin" in top-right corner
- ✅ Backend connection status shows "Connected"

#### Test Case 1.2: Invalid Credentials
**Steps:**
1. Try login with wrong username/password combinations
2. Verify error messages are displayed
3. Ensure no redirect occurs

**Expected Results:**
- ✅ Error message displayed
- ✅ Remains on login page
- ✅ No authentication tokens created

### 2. API Key Authentication

#### Test Case 2.1: Get Current API Key
**Steps:**
1. GET request to: `http://localhost:8002/api/v1/auth/debug/admin-api-key`
2. Note the `api_key` value from response

#### Test Case 2.2: API Key Login (Browser)
**Steps:**
1. Navigate to login page
2. Click "API Key" tab
3. Enter the API key from Test Case 2.1
4. Click "Sign In"

**Expected Results:**
- ✅ Successful authentication
- ✅ Redirect to dashboard
- ✅ Same functionality as username/password login

#### Test Case 2.3: API Key Login (Direct API)
**Steps:**
```bash
curl -X POST http://localhost:8002/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"api_key": "YOUR_API_KEY_HERE"}'
```

**Expected Results:**
- ✅ Returns JWT tokens
- ✅ HTTP 200 status code

### 3. Session Management

#### Test Case 3.1: Session Persistence
**Steps:**
1. Login using any method
2. Close browser completely
3. Reopen browser and navigate to http://localhost:3001

**Expected Results:**
- ✅ Automatically logged in (no login page shown)
- ✅ Dashboard loads directly
- ✅ User session maintained

#### Test Case 3.2: Token Refresh
**Steps:**
1. Login and monitor network traffic
2. Wait and observe automatic refresh requests
3. Verify tokens are refreshed before expiration

**Expected Results:**
- ✅ Automatic token refresh occurs
- ✅ No user interruption
- ✅ Session remains active

#### Test Case 3.3: Logout Functionality
**Steps:**
1. Login to dashboard
2. Click user dropdown menu (admin)
3. Select "Logout"

**Expected Results:**
- ✅ Redirect to login page
- ✅ Session tokens invalidated
- ✅ Cannot access dashboard without re-login

### 4. Route Protection

#### Test Case 4.1: Protected Route Access (Unauthenticated)
**Steps:**
1. Open incognito/private browser window
2. Navigate directly to http://localhost:3001/dashboard

**Expected Results:**
- ✅ Automatic redirect to login page
- ✅ Cannot access dashboard content

#### Test Case 4.2: Protected Route Access (Authenticated)
**Steps:**
1. Login normally
2. Try accessing various routes
3. Verify dashboard functionality

**Expected Results:**
- ✅ Dashboard accessible
- ✅ All protected features work
- ✅ No unauthorized access allowed

### 5. Security Testing

#### Test Case 5.1: Invalid API Key
**Steps:**
```bash
curl -X POST http://localhost:8002/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"api_key": "invalid_key_123"}'
```

**Expected Results:**
- ✅ HTTP 401 Unauthorized
- ✅ Error message: "Invalid API key"

#### Test Case 5.2: Malformed Requests
**Steps:**
1. Send requests with missing fields
2. Send requests with invalid JSON
3. Test with special characters

**Expected Results:**
- ✅ Appropriate error responses
- ✅ No system crashes
- ✅ Security headers present

## API Endpoints Reference

### Authentication Endpoints

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/api/v1/auth/login` | POST | User login | No |
| `/api/v1/auth/logout` | POST | User logout | Yes |
| `/api/v1/auth/refresh` | POST | Token refresh | No* |
| `/api/v1/auth/me` | GET | User info | Yes |
| `/api/v1/auth/validate` | GET | Token validation | Yes |

*Requires refresh token in cookie or request body

### Request/Response Examples

#### Login Request (Username/Password)
```json
{
  "username": "admin",
  "password": "admin123"
}
```

#### Login Request (API Key)
```json
{
  "api_key": "Yo5uWJVyCTCVwoY2_8kBk49j5PBKdY-ts7ygKUMAFHw"
}
```

#### Login Response
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

## Browser Compatibility

Test the application on:
- ✅ Chrome (latest)
- ✅ Firefox (latest) 
- ✅ Safari (latest)
- ✅ Edge (latest)

## Performance Testing

### Load Testing
- Test concurrent logins
- Verify token refresh under load
- Monitor memory usage during extended sessions

### Security Testing
- Test for XSS vulnerabilities
- Verify CSRF protection
- Test session hijacking prevention

## Known Limitations

1. **Debug Endpoint**: The `/api/v1/auth/debug/admin-api-key` endpoint is for testing only and should be removed in production
2. **In-Memory Database**: User data is stored in memory and resets on server restart
3. **Single User**: Currently supports one admin user account

## Acceptance Criteria Verification

- ✅ **Authentication System**: Both username/password and API key methods implemented
- ✅ **Session Management**: JWT token-based with automatic refresh
- ✅ **Route Protection**: All sensitive routes protected
- ✅ **Session Persistence**: Works across browser restarts
- ✅ **Session Handling**: Automatic refresh prevents unexpected logouts

## Reporting Issues

When reporting issues, please include:
1. Test case number and description
2. Steps to reproduce
3. Expected vs actual results
4. Browser and version
5. Console errors (if any)
6. Network request/response details

## Final Verification Checklist

Before marking Story 1.4 as complete:

- [ ] All test cases pass
- [ ] No console errors in browser
- [ ] API endpoints respond correctly
- [ ] Session persistence works
- [ ] Route protection functional
- [ ] Both authentication methods work
- [ ] Logout functionality works
- [ ] Security measures in place
- [ ] Documentation complete
- [ ] Debug endpoints removed (production)

---

**Story 1.4 Status**: Ready for QA Testing
**Last Updated**: August 16, 2025
**Tested By**: Development Team