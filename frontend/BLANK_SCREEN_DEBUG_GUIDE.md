# React Blank Screen Debugging Guide

## ⚠️ RECURRING ISSUE: Frontend Blank Screen 

This is a common problem that happens when React components crash during rendering. Here's the systematic approach to debug and fix it.

## 🔍 Debugging Steps

### 1. Use Playwright to Get Real Browser Information
```bash
npx playwright test --headed --project=chromium tests/e2e/chart-interactions.spec.ts -g "should display" --timeout=15000
```

**What to look for:**
- Browser console errors (BROWSER: lines in output)
- Specific component crash locations
- React error boundaries triggering

### 2. Check TypeScript Compilation Errors
```bash
docker exec nautilus-frontend npx tsc --noEmit --skipLibCheck 2>&1 | head -20
```

**Common causes:**
- Missing import statements
- Incorrect prop types
- Missing dependencies

### 3. Check Backend Connectivity
```bash
curl -s http://localhost:8001/health
```

**If backend is down:**
- `docker-compose restart backend`
- Check backend logs: `docker-compose logs backend --tail=20`

### 4. Check Frontend Container Status
```bash
docker-compose logs frontend --tail=20
```

**Look for:**
- Vite startup messages
- HMR (Hot Module Reload) errors
- Build failures

## 🐛 Common Root Causes & Fixes

### 1. Missing Icon Imports
**Symptom:** Dashboard crashes when rendering tabs
**Fix:** Check all imported icons are correctly named

**Common problematic icons:**
```typescript
// ❌ WRONG ICONS - These don't exist in Ant Design:
import { 
  ExclamationTriangleOutlined, // Use ExclamationCircleOutlined
  CompareOutlined,             // Use DiffOutlined or SwapOutlined  
  ThunderboltOutlinedd,        // Typo - use ThunderboltOutlined
  WifiOff                      // Use WifiOutlined with conditional logic
} from '@ant-design/icons'

// ✅ CORRECT ICONS:
import { 
  ExclamationCircleOutlined,
  DiffOutlined, 
  ThunderboltOutlined,
  WifiOutlined
} from '@ant-design/icons'
```

**Quick test for icon issues:**
```bash
node -e "const icons = require('@ant-design/icons'); console.log(icons.ExclamationTriangleOutlined ? 'exists' : '❌ MISSING');"
```

### 2. Incompatible Hook Interfaces
**Symptom:** Components crash when using hooks
**Fix:** Ensure hook return types match component expectations
```typescript
// Check that all used properties exist in hook return type
const { connectionStatus, getLatestMessageByTopic } = useMessageBus()
```

### 3. Import Path Issues
**Symptom:** "Cannot resolve module" errors
**Fix:** Use correct relative import paths
```typescript
// ❌ Wrong
import { Component } from '../../../components/Component'

// ✅ Correct (check actual file structure)
import { Component } from '../components/Component'
```

### 4. Missing Environment Variables
**Symptom:** Network requests fail, blank screen
**Fix:** Check vite.config.ts proxy configuration and environment variables

### 5. Component Render Errors
**Symptom:** React error boundary messages in console
**Fix:** Wrap problematic components in ErrorBoundary or fix the underlying issue

## 🚀 Quick Fix Pattern

1. **Isolate the problem** - Comment out recently added components
2. **Test incrementally** - Add components back one by one
3. **Use error boundaries** - Wrap new components in ErrorBoundary
4. **Check Playwright output** - Always verify with actual browser testing

## 📝 Emergency Recovery Steps

If Dashboard is completely broken:

1. **Revert recent changes:**
```bash
git stash  # Save current work
git checkout HEAD~1 -- src/pages/Dashboard.tsx  # Revert Dashboard
```

2. **Test basic functionality:**
```bash
npx playwright test -g "should display" --timeout=10000
```

3. **Add changes back incrementally:**
- One component at a time
- Test after each addition
- Use ErrorBoundary wrappers

## 🔧 Prevention

1. **Always test with Playwright** after making component changes
2. **Use ErrorBoundary** around new components
3. **Check TypeScript compilation** before committing
4. **Test both dev and build modes**

## 📋 Debugging Checklist

- [ ] Playwright test shows actual browser state
- [ ] TypeScript compilation passes
- [ ] Backend health check passes  
- [ ] Frontend container logs show no errors
- [ ] All imports are correct
- [ ] Environment variables are set
- [ ] Error boundaries are in place

## ⚡ Last Known Working State

When Dashboard was working:
- Date: [Fill in when fixed]
- Key components: [List working components]
- Known issues: [Document any ongoing problems]

---

**Remember:** The blank screen is usually React failing to render due to JavaScript errors. Playwright gives you the real browser console output to debug these issues effectively.