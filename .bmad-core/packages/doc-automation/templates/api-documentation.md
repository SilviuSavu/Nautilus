# API Documentation Template

**Status**: 📋 Template  
**Category**: API Documentation  
**BMAD Package**: doc-automation v1.0.0

## Overview

**Brief API description in bold text explaining the core functionality and purpose.**

## Quick Reference

### Base Information
- **Base URL**: `https://api.example.com/v1`
- **Authentication**: API Key required
- **Rate Limits**: 1000 requests/hour
- **Response Format**: JSON
- **Status**: ✅ Production Ready

### Key Endpoints
- `GET /health` - System health check
- `POST /auth/login` - User authentication
- `GET /users/{id}` - Retrieve user information
- `POST /users` - Create new user

## Authentication

### API Key Authentication
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.example.com/v1/endpoint
```

### OAuth 2.0 Flow
```bash
# Step 1: Authorization Code
https://api.example.com/oauth/authorize?client_id=CLIENT_ID&response_type=code&redirect_uri=REDIRECT_URI

# Step 2: Access Token
curl -X POST https://api.example.com/oauth/token \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "grant_type=authorization_code&code=AUTH_CODE&client_id=CLIENT_ID&client_secret=CLIENT_SECRET"
```

## Core Endpoints

### Health Check
**Endpoint**: `GET /health`  
**Purpose**: System status and availability check  
**Authentication**: None required

#### Request
```bash
curl https://api.example.com/v1/health
```

#### Response
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-08-26T01:30:00Z",
  "services": {
    "database": "healthy",
    "cache": "healthy",
    "external_apis": "healthy"
  }
}
```

### User Management

#### Get User
**Endpoint**: `GET /users/{id}`  
**Purpose**: Retrieve user information by ID  
**Authentication**: API Key required

##### Request
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.example.com/v1/users/123
```

##### Response
```json
{
  "id": 123,
  "username": "john_doe",
  "email": "john@example.com",
  "created_at": "2025-01-01T00:00:00Z",
  "last_active": "2025-08-26T01:30:00Z",
  "profile": {
    "first_name": "John",
    "last_name": "Doe",
    "avatar_url": "https://example.com/avatars/123.jpg"
  }
}
```

##### Error Responses
```json
// User not found (404)
{
  "error": {
    "code": "USER_NOT_FOUND",
    "message": "User with ID 123 not found",
    "timestamp": "2025-08-26T01:30:00Z"
  }
}

// Unauthorized (401)
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing API key",
    "timestamp": "2025-08-26T01:30:00Z"
  }
}
```

#### Create User
**Endpoint**: `POST /users`  
**Purpose**: Create a new user account  
**Authentication**: API Key required

##### Request
```bash
curl -X POST https://api.example.com/v1/users \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "jane_doe",
       "email": "jane@example.com",
       "password": "secure_password_123",
       "profile": {
         "first_name": "Jane",
         "last_name": "Doe"
       }
     }'
```

##### Response
```json
{
  "id": 124,
  "username": "jane_doe",
  "email": "jane@example.com",
  "created_at": "2025-08-26T01:30:00Z",
  "profile": {
    "first_name": "Jane",
    "last_name": "Doe",
    "avatar_url": null
  }
}
```

## Request/Response Patterns

### Standard Response Structure
```json
{
  "data": { ... },           // Main response data
  "meta": {                  // Metadata about the response
    "timestamp": "2025-08-26T01:30:00Z",
    "request_id": "req_123456789",
    "api_version": "v1"
  },
  "pagination": {            // For paginated responses
    "page": 1,
    "per_page": 20,
    "total_pages": 5,
    "total_count": 100
  }
}
```

### Error Response Structure
```json
{
  "error": {
    "code": "ERROR_CODE",      // Machine-readable error code
    "message": "Human readable error message",
    "details": { ... },        // Additional error context
    "timestamp": "2025-08-26T01:30:00Z",
    "request_id": "req_123456789"
  }
}
```

## HTTP Status Codes

| Code | Meaning | Usage |
|------|---------|--------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request succeeded, no response body |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required or failed |
| 403 | Forbidden | Access denied |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Service temporarily unavailable |

## Rate Limiting

### Limits
- **Standard Tier**: 1,000 requests/hour
- **Premium Tier**: 10,000 requests/hour
- **Enterprise Tier**: 100,000 requests/hour

### Headers
Response includes rate limit information:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 3600
```

### Rate Limit Exceeded Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit of 1000 requests per hour exceeded",
    "retry_after": 3600,
    "timestamp": "2025-08-26T01:30:00Z"
  }
}
```

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Response Time (95th percentile) | < 200ms | 150ms |
| Availability | 99.9% | 99.95% |
| Error Rate | < 0.1% | 0.05% |
| Rate Limit Accuracy | 100% | 100% |

## SDK Examples

### JavaScript/Node.js
```javascript
const ApiClient = require('./api-client');

const client = new ApiClient({
  apiKey: 'YOUR_API_KEY',
  baseUrl: 'https://api.example.com/v1'
});

// Get user
const user = await client.users.get(123);
console.log(user.username);

// Create user
const newUser = await client.users.create({
  username: 'jane_doe',
  email: 'jane@example.com',
  password: 'secure_password_123'
});
```

### Python
```python
from api_client import ApiClient

client = ApiClient(
    api_key='YOUR_API_KEY',
    base_url='https://api.example.com/v1'
)

# Get user
user = client.users.get(123)
print(user.username)

# Create user
new_user = client.users.create({
    'username': 'jane_doe',
    'email': 'jane@example.com',
    'password': 'secure_password_123'
})
```

## Testing

### Postman Collection
Download the complete Postman collection: [API Collection.json](./postman/collection.json)

### Test Environment
- **Base URL**: `https://api-staging.example.com/v1`
- **Test API Key**: `test_key_123456789`
- **Rate Limits**: Same as production

### Sample Test Cases
```bash
# Health check
curl https://api-staging.example.com/v1/health

# Authentication test
curl -H "Authorization: Bearer test_key_123456789" \
     https://api-staging.example.com/v1/users/1

# Error handling test
curl https://api-staging.example.com/v1/users/999999
```

## Troubleshooting

### Common Issues

#### 401 Unauthorized
- **Cause**: Invalid or missing API key
- **Solution**: Verify API key is correct and included in Authorization header
- **Example**: `Authorization: Bearer YOUR_ACTUAL_API_KEY`

#### 429 Rate Limit Exceeded
- **Cause**: Too many requests in time window
- **Solution**: Implement exponential backoff and respect rate limit headers
- **Code Example**:
```javascript
if (response.status === 429) {
  const retryAfter = response.headers['retry-after'];
  await sleep(retryAfter * 1000);
  // Retry request
}
```

#### 500 Internal Server Error
- **Cause**: Server-side issue
- **Solution**: Contact support with request ID from error response
- **Monitoring**: Check status page at https://status.example.com

### Support Channels
- **Documentation**: https://docs.example.com/api
- **Support Email**: api-support@example.com
- **Status Page**: https://status.example.com
- **Community Forum**: https://community.example.com

## Changelog

### v1.0.0 (2025-08-26)
- Initial API release
- User management endpoints
- Authentication system
- Rate limiting implementation

---

**Generated by**: BMAD Documentation Template System  
**Template**: api-documentation.md  
**Version**: 1.0.0  
**Last Updated**: $(date)

## Template Usage

This template should be customized by replacing:
- `example.com` with your actual API domain
- `YOUR_API_KEY` with actual authentication details
- Endpoint examples with your specific API endpoints
- Performance metrics with your actual targets
- Support information with your contact details

### BMAD Commands for API Documentation

```bash
# Apply this template to a new API doc
bmad apply template api-documentation target=docs/api/new-service.md

# Validate API documentation standards
bmad run check-doc-health include_patterns="['docs/api/**']"

# Generate API cross-references
bmad run generate-doc-sitemap include_patterns="['docs/api/**']" group_by=category
```