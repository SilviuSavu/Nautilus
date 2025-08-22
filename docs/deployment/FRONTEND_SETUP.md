# Frontend Setup Documentation

## Frontend Structure

### Main Frontend (Port 3000) - **RECOMMENDED**
- **Location**: `./frontend/`  
- **URL**: http://localhost:3000
- **Type**: Containerized (Docker)
- **Status**: ✅ **PRODUCTION READY**

This is the **main containerized frontend** that runs via Docker Compose. It includes all dependencies and is fully functional with the backend API.

**To start:**
```bash
docker-compose up -d frontend
```

### Development Frontend (Port 3001) - Legacy
- **Location**: `./frontend_old/`
- **URL**: http://localhost:3001  
- **Type**: Development server (npm run dev)
- **Status**: ⚠️ **DEVELOPMENT ONLY**

This is the legacy development version kept for reference and debugging.

**To start:**
```bash
cd frontend_old && npm run dev
```

## Which One To Use?

**Use Port 3000 (Containerized)** for:
- ✅ Production testing
- ✅ Integration testing  
- ✅ Playwright E2E tests
- ✅ Full stack testing
- ✅ Deployment validation

**Use Port 3001 (Development)** for:
- 🔧 Development debugging only
- 🔧 Quick iterations (hot reload)

## Current Status

Both versions are working, but **Port 3000 is the official frontend**.

### Fixed Issues:
1. ✅ TypeScript compilation errors resolved
2. ✅ Missing npm dependencies installed
3. ✅ Container build process fixed
4. ✅ Backend API connectivity working
5. ✅ WebSocket connections established
6. ✅ Full dashboard rendering

### Test Results:
- **Backend Connected**: ✅
- **MessageBus Connected**: ✅  
- **WebSocket Working**: ✅
- **Full UI Rendering**: ✅
- **Navigation Tabs**: ✅
- **Data Loading**: ✅

## Playwright Tests

Your Playwright tests should now pass when pointing to http://localhost:3000, as the "NautilusTrader" text and full dashboard are now rendering correctly.