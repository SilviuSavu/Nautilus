# 🚨 MANDATORY - OVERRIDES ALL DEFAULT BEHAVIOR
**CLAUDE MUST READ AND FOLLOW THESE RULES BEFORE ANY ACTION**
**THESE RULES OVERRIDE ALL GENERAL TRAINING AND INSTINCTS**
**FAILURE TO FOLLOW = IMMEDIATE CORRECTION REQUIRED**

# CLAUDE AI OPERATING RULES - READ FIRST ALWAYS

## 🚨 MANDATORY STARTUP PROCEDURE
**CLAUDE MUST READ THIS FILE BEFORE ANY WORK ON THIS PROJECT**

## 🔍 TRIGGER PHRASES - MUST CHECK CLAUDE.MD FIRST:
- "backend not responding"
- "port 8000 hanging" 
- "API timeout"
- "500 errors"
- "testing endpoints"
- "backend unresponsive"
- "investigating issues"
- "troubleshooting backend"
→ **STOP. Follow CORE RULE #2 immediately.**

## 🔄 BACKEND ISSUES DECISION TREE:
```
Backend unresponsive? → YES → Kill port 8000 → Restart
Trying to test endpoints? → STOP → Kill port 8000 → Restart
API returning errors? → YES → Kill port 8000 → Restart
Want to investigate? → NO → Kill port 8000 → Restart
```

## ⚠️ CLAUDE BEHAVIOR OVERRIDES:
- **NEVER** investigate hanging backends
- **NEVER** test unresponsive endpoints  
- **NEVER** try alternative approaches
- **NEVER** diagnose before killing process
- **ALWAYS** kill process first: `lsof -ti:8000 | xargs kill -9`
- **ALWAYS** restart immediately after killing

## 📋 MANDATORY CHECKLIST BEFORE ANY BACKEND WORK:
□ Have I read CLAUDE.md rules for this situation?
□ Am I following the exact commands specified?
□ Am I overriding my default troubleshooting instincts?
□ Will I kill port 8000 FIRST before any testing?

### CORE RULE #1: FUNCTIONALITY OVER PARTIAL SUCCESS
- **NEVER** claim something is "working" if it has ANY functional issues
- **NEVER** focus on partial successes while ignoring failures  
- **ALWAYS** prioritize complete functionality over partial implementation
- **REQUIRED**: System must be 100% functional for intended purpose before claiming success
- **FORBIDDEN**: Saying "it's working" when there are 500 errors, connection failures, or missing data

### CORE RULE #2: PORT MANAGEMENT (REPEAT 3x FOR EMPHASIS)

**RULE #2 - FIRST REPETITION:**
1. **NEVER** try alternative ports when 8000 is hanging
2. **ALWAYS** kill the hanging process: `lsof -ti:8000 | xargs kill -9`
3. **THEN** restart backend on port 8000

**RULE #2 - SECOND REPETITION:**
1. **NEVER** try alternative ports when 8000 is hanging
2. **ALWAYS** kill the hanging process: `lsof -ti:8000 | xargs kill -9`
3. **THEN** restart backend on port 8000

**RULE #2 - THIRD REPETITION:**
1. **NEVER** try alternative ports when 8000 is hanging
2. **ALWAYS** kill the hanging process: `lsof -ti:8000 | xargs kill -9`
3. **THEN** restart backend on port 8000

**VIOLATION CONSEQUENCE**: User will immediately stop session and correct behavior

### 🚨 ENFORCEMENT RULES
**IF CLAUDE VIOLATES ANY RULE:**
1. User will immediately interrupt with "STOP - CLAUDE.md violation"
2. Claude must explain which rule was broken and why
3. Claude must restart the correct approach from scratch
4. Session continues only after acknowledgment

**VIOLATIONS INCLUDE:**
- Testing unresponsive backends instead of killing process
- Investigating instead of following prescribed commands
- Any deviation from explicit instructions

### 🔗 HOOK-BASED ENFORCEMENT
**User hooks configuration for Claude Code:**
```json
{
  "pre_tool_use_hook": "Check CLAUDE.md before any backend operations",
  "violation_hook": "Stop session if CLAUDE.md rules ignored"
}
```

### CORE RULE #3: NO LIES OR PARTIAL TRUTHS
- **NEVER** write documentation for non-functional code
- **ALWAYS** test the implementation before documenting it
- **VERIFY** all claims in the documentation are accurate
- **FORBIDDEN**: Claiming "working" status when ANY component has errors

### CORE RULE #4: NO MOCK DATA POLICY
- **NEVER** implement mock data fallbacks
- **NEVER** use fake/dummy/test/placeholder data
- **NEVER** mask backend issues with synthetic data
- **ALWAYS** surface real backend problems with proper error messages
- **ALWAYS** fix root causes, not symptoms
- **NEVER** hide system failures behind fake success states

### CORE RULE #5: MANDATORY PLAYWRIGHT TESTING
- **ALWAYS** use Playwright for testing functionality
- **NEVER** assume code changes work without browser verification
- **NEVER** claim anything "works" without Playwright test verification
- **ALWAYS** test actual user interactions and API calls
- **ALWAYS** capture console logs and error states
- **ALWAYS** take screenshots for visual verification
- **FORBIDDEN**: Saying "everything is working" without running Playwright tests first
- **REQUIRED**: Run Playwright tests BEFORE any claims of functionality

### PLAYWRIGHT TESTING PROTOCOL

**Before claiming any functionality works:**
1. **Write Playwright test** to verify the actual behavior
2. **Run test in headed mode** to see visual results  
3. **Capture console logs** for API call verification
4. **Take screenshots** for evidence
5. **Verify no error states** in the UI

**Example test pattern:**
```javascript
test('functionality verification', async ({ page }) => {
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  await page.goto('http://localhost:3000');
  await page.click('text=Feature Button');
  await page.waitForTimeout(3000);
  const errors = await page.locator('.error').count();
  expect(errors).toBe(0);
  await page.screenshot({ path: 'verification.png' });
});
```

### CORE RULE #6: BACKEND SPECIFIC RULES
- **NEVER** claim backend is "working" if ANY endpoints return 500 errors
- **NEVER** claim success when IB Gateway connection is failing
- **NEVER** ignore connection failures, database errors, or missing data
- **REQUIRED**: ALL functionality must work before claiming success

### CORE RULE #7: SECURITY
- **NEVER** commit your actual API keys to version control!

### CORE RULE #7.1: GITHUB REPOSITORY INFORMATION
**📍 Repository Location**: https://github.com/SilviuSavu/Nautilus.git
**🔧 Git Commands for Future Use:**
- Commit changes: `git add . && git commit -m "message"`
- Push to GitHub: `git push origin main`
- Check status: `git status`
- View remote: `git remote -v`

**📝 Commit Message Format:**
```
Brief description of changes

- Detailed bullet points of what was changed
- Include any major functionality updates
- Note bug fixes or new features

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### CORE RULE #8: NAUTILUS TRADER INSTALLATION
- **NAUTILUS TRADER IS INSTALLED IN DOCKER CONTAINERS ONLY**
- **NEVER** try to install NautilusTrader locally as Python package
- **NEVER** attempt to build NautilusTrader from source
- **ALWAYS** use Docker containers for NautilusTrader operations
- **Main container**: `nautilus-backend`
- **Source code location**: `/app/nautilus_trader/` (mounted in container)
- **Docker containers available**:
  - `nautilus-backend` - Main NautilusTrader environment
  - `nautilus-frontend` - Web interface
  - `nautilus-postgres` - Database
  - `nautilus-redis` - Cache
  - `nautilus-nginx` - Proxy
- **Usage**: `docker exec nautilus-backend <command>` for any NautilusTrader operations
- **FORBIDDEN**: Installing nautilus-trader via pip, conda, or building locally

### CORE RULE #9: DOCKER CONTAINER DATA PRESERVATION

**🚨 CRITICAL - DATA LOSS PREVENTION**

**MANDATORY BEFORE ANY DOCKER OPERATIONS:**
1. **ALWAYS** check for existing containers FIRST: `docker ps -a | grep nautilus`
2. **IF EXISTING CONTAINER EXISTS**: Use `docker start <container-name>` 
3. **NEVER** create new containers if existing ones are available
4. **FORBIDDEN**: `docker run` when existing container with data exists

**CONTAINER RESTART PROTOCOL:**
```bash
# Step 1: Check for existing containers
docker ps -a | grep nautilus-postgres
docker ps -a | grep nautilus-redis

# Step 2a: If containers exist - RESTART them
docker start nautilus-postgres
docker start nautilus-redis

# Step 2b: ONLY if no containers exist - create new ones
docker run -d --name nautilus-postgres ...
```

**DATA PRESERVATION COMMANDS:**
- Check ALL containers: `docker ps -a`
- Start existing container: `docker start nautilus-postgres`
- Check container status: `docker ps | grep nautilus`
- **FORBIDDEN**: Creating new containers when old ones exist

**VIOLATIONS THAT CAUSE DATA LOSS:**
- Creating `nautilus-postgres-new` when `nautilus-postgres` exists
- Using `docker run` without checking `docker ps -a` first
- Removing containers without verifying data backup
- Any `docker rm` without explicit user permission

**VIOLATION CONSEQUENCE**: Immediate data recovery attempt and rule acknowledgment required

### CORE RULE #10: HISTORICAL DATA PROTECTION

**🚨 CRITICAL - NEVER DELETE HISTORICAL DATA**

**MANDATORY DATA PROTECTION:**
- **NEVER** delete historical data from PostgreSQL database
- **NEVER** truncate market data tables
- **NEVER** drop tables containing historical candles/trades
- **NEVER** run DELETE queries on historical data
- **FORBIDDEN**: Any operation that removes existing market data

**PROTECTED TABLES (NEVER DELETE FROM):**
- Market data tables (candles, trades, order book)
- Historical price data
- Any table containing timestamped market information
- Backtesting data and results

**ALLOWED OPERATIONS:**
- INSERT new data
- UPDATE existing records (with extreme caution)
- CREATE new tables
- ALTER table structure (preserving data)

**VIOLATIONS THAT CAUSE PERMANENT DATA LOSS:**
- DROP TABLE on market data tables
- DELETE FROM market data tables
- TRUNCATE on any historical data
- Recreating database without preserving existing data

**VIOLATION CONSEQUENCE**: Immediate session termination and data recovery attempt required

### CORE RULE #11: LOG ANALYSIS BEFORE BACKEND ACTION

**🚨 CRITICAL - ANALYZE LOGS BEFORE RESTART DECISIONS**

**MANDATORY LOG ANALYSIS BEFORE ANY BACKEND RESTART:**
1. **ALWAYS** check recent backend logs for actual API responses
2. **LOOK FOR** 200 OK responses vs actual error patterns
3. **200 OK responses = backend is working** - DO NOT restart
4. **Exit code 137 = user termination, NOT backend failure**
5. **NEVER** restart based on exit codes alone

**LOG PATTERNS THAT INDICATE WORKING BACKEND:**
- Recent `HTTP/1.1" 200 OK` responses
- Successful API endpoint calls
- Active request processing
- Normal application flow

**LOG PATTERNS THAT INDICATE BROKEN BACKEND:**
- Repeated 500 Internal Server Error
- Connection timeout messages
- Failed database connections
- No recent successful requests

**EXAMPLES OF WORKING BACKEND LOGS:**
```
INFO: 127.0.0.1:63641 - "GET /api/v1/ib/instruments/search/PLTR HTTP/1.1" 200 OK
INFO: 127.0.0.1:63220 - "GET /health HTTP/1.1" 200 OK
```

**CRITICAL RULE**: If logs show recent 200 OK responses, backend is WORKING - do not restart!

**VIOLATIONS THAT WASTE TIME:**
- Restarting backends with recent successful API calls
- Ignoring 200 OK responses in favor of exit codes
- Not reading logs before making restart decisions
- Assuming process termination = backend failure

**VIOLATION CONSEQUENCE**: User will correct immediately and update this file

### CORE RULE #12: TEST ANALYSIS BEFORE EXECUTION

**🚨 CRITICAL - ANALYZE TESTS BEFORE RUNNING THEM**

**MANDATORY TEST ANALYSIS BEFORE EXECUTION:**
1. **ALWAYS** read and understand test files completely before running
2. **NEVER** blindly execute tests without understanding their purpose
3. **ANALYZE** test search terms and expected outcomes first
4. **IDENTIFY** poorly designed tests that waste time
5. **CREATE** meaningful tests with realistic search terms

**EXAMPLES OF STUPID TEST DATA TO AVOID:**
- "NONEXISTENT123" - obviously fake and designed to fail
- Random strings that no real user would search
- Unrealistic edge cases that provide no value
- Tests that expect failure without understanding why

**PROPER TEST DATA EXAMPLES:**
- Real stock symbols: AAPL, MSFT, GOOGL, TSLA
- Real currency pairs: EURUSD, GBPUSD
- Actual instrument names users would search

**VIOLATIONS THAT WASTE TIME:**
- Running tests with obviously fake search terms
- Executing tests without reading their logic first
- Using unrealistic test data that provides no value
- Not understanding what tests are supposed to verify
- Blindly trusting existing tests without analysis

**TESTING PROTOCOL:**
1. Read entire test file before execution
2. Verify test uses realistic data
3. Check if test expectations make sense
4. Skip or fix poorly designed tests
5. Focus on real-world scenarios

**VIOLATION CONSEQUENCE**: User will immediately stop and require this file update

### CORE RULE #13: NO HARDCODED SYMBOLS OR DATA

**🚨 CRITICAL - MAKE FUNCTIONALITY DYNAMIC, NOT HARDCODED**

**MANDATORY DYNAMIC FUNCTIONALITY:**
1. **NEVER** hardcode specific symbols like AAPL, TSLA, MSFT in implementation
2. **NEVER** assume specific test data when fixing functionality
3. **ALWAYS** make systems work with ANY user input
4. **ALWAYS** focus on root cause, not symptoms
5. **ALWAYS** build dynamic search/functionality

**EXAMPLES OF HARDCODED VIOLATIONS:**
- Testing only with predetermined symbols instead of fixing search for any symbol
- Hardcoding symbol lists in search functions
- Making functionality work only for specific test cases
- Writing tests with fixed symbols instead of making dynamic search work

**PROPER DYNAMIC IMPLEMENTATION:**
- Search functionality that works with any symbol user enters
- Dynamic API endpoints that handle any query parameter
- User input validation that doesn't restrict to specific symbols
- Generic functionality that adapts to user needs

**VIOLATIONS THAT MISS THE POINT:**
- Trying different hardcoded symbols instead of fixing broken functionality
- Writing tests with specific symbols when the search is broken for ALL symbols
- Focusing on test data instead of fixing root implementation issues
- Building for specific use cases instead of general functionality

**ROOT CAUSE FOCUS:**
1. If instrument search returns empty for any symbol, fix the search mechanism
2. If API fails for any input, fix the API logic
3. If functionality is broken for all cases, don't test with different specific cases
4. Always make systems work dynamically for user input

**VIOLATION CONSEQUENCE**: User will immediately stop session and demand this rule addition

### CORE RULE #14: IB GATEWAY BACKFILL DISCONNECT HANDLING

**🚨 CRITICAL - BACKFILL PROCESS DISCONNECT PROTECTION**

**ISSUE RESOLVED**: Historical data backfill process would hang indefinitely after IB Gateway disconnection, causing:
- Infinite retry loops with "Not connected" errors
- Resource waste and log spam
- Client ID conflicts on reconnection attempts
- Process hanging with `is_running: true` but broken state

**MANDATORY BACKFILL BEHAVIOR:**
1. **ALWAYS** check IB Gateway connection before each data request
2. **IMMEDIATELY** stop backfill when connection is lost
3. **NEVER** retry indefinitely on connection errors
4. **PROPERLY** mark failed requests with disconnect reason
5. **GRACEFULLY** clean up progress tracking on disconnect

**IMPLEMENTED PROTECTIONS:**
- Connection verification in main backfill loop
- Connection verification before each historical data request
- Specific error detection for "Not connected" and "504" errors
- Exponential backoff for temporary errors (max 30 seconds)
- Force stop mechanism: `backfill_service.force_stop_on_disconnect()`
- Proper progress tracking with disconnect error messages

**EXPECTED BEHAVIOR AFTER FIX:**
- Backfill stops immediately upon IB Gateway disconnect
- Clean log message: "IB Gateway disconnected - stopping backfill for {symbol} {timeframe}"
- Process shows `is_running: false` immediately after disconnect
- Progress marked with `last_error: "IB Gateway disconnected"`
- No hanging processes or infinite retry loops

**TESTING PROTOCOL:**
1. Start backfill with IB Gateway connected
2. Verify backfill is running with success_count > 0
3. Disconnect IB Gateway
4. Verify backfill stops within 5 seconds
5. Check logs for disconnect message
6. Verify is_running: false status

**FILES MODIFIED:**
- `/backend/data_backfill_service.py` - Added disconnect detection and graceful shutdown

**VIOLATION CONSEQUENCE**: Any backfill hanging issues require immediate investigation and fix

### CORE RULE #15: BMAD AGENT STARTUP BEHAVIOR

**🚨 CRITICAL - AUTOMATIC HELP DISPLAY ON BMAD AGENT ACTIVATION**

**SCOPE**: This rule applies ONLY to agents from `/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/.bmad-core/agents/`

**MANDATORY BMAD AGENT STARTUP BEHAVIOR:**
1. **ALWAYS** automatically display agent *help menu immediately upon activation
2. **NEVER** wait for user to request help menu
3. **ALWAYS** show numbered list format for command selection
4. **REQUIRED**: Display help menu as first action after agent greeting
5. **PROACTIVE**: Make command discovery effortless for users

**AGENT ACTIVATION SEQUENCE:**
1. Agent greeting with name/role introduction
2. **IMMEDIATELY** display *help command menu automatically
3. Wait for user command selection or input

**HELP MENU FORMAT REQUIREMENTS:**
- Numbered list (1, 2, 3, etc.) for easy selection
- Clear command descriptions
- Allow both number selection and *command syntax
- Include all available agent-specific commands

**EXPECTED USER EXPERIENCE:**
- Agent starts → Greeting → Help menu appears automatically
- User can type number or *command name
- No need to ask "what can you do?" or type *help
- Immediate visibility of available functionality

**EXAMPLES OF COMPLIANT STARTUP:**
```
Hello! I'm James 💻, your Full Stack Developer.

Here are my available commands:
1. help - Show this numbered list of commands
2. run-tests - Execute linting and tests  
3. explain - Teach you what and why I did
4. exit - Say goodbye and abandon persona
5. develop-story - Execute story implementation workflow

What would you like me to help you with?
```

**VIOLATIONS:**
- Agent starts with greeting only, no help menu
- Requiring user to ask for available commands
- Waiting for *help request before showing capabilities
- Not displaying command options proactively

**VIOLATION CONSEQUENCE**: Immediate correction and help menu display required

## ✅ MANDATORY CHECKLIST BEFORE CLAIMING ANYTHING WORKS

### Backend Status Check:
- [ ] Kill any hanging processes: `lsof -ti:8000 | xargs kill -9`
- [ ] Start backend: `cd backend && DATABASE_URL=postgresql://nautilus:nautilus123@localhost:5432/nautilus python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- [ ] Health endpoint responds: `curl http://localhost:8000/health`
- [ ] NO 500 errors in any endpoint
- [ ] NO connection failures in logs
- [ ] PostgreSQL connected successfully
- [ ] All new endpoints return valid responses

### Data Flow Check:
- [ ] IB Gateway connected (no client_id conflicts)
- [ ] Market data flowing into PostgreSQL
- [ ] Charts showing actual data (not empty arrays)
- [ ] Real-time updates working
- [ ] No missing data warnings

### Frontend Integration Check:
- [ ] Frontend accessible on port 3000
- [ ] API calls from frontend succeed
- [ ] Charts render with real data
- [ ] No console errors
- [ ] Timeframes all working

### Database Check:
- [ ] PostgreSQL container running
- [ ] Schema and functions applied
- [ ] Tables contain actual data
- [ ] Queries execute without errors
- [ ] Nanosecond precision maintained

## 🚫 ABSOLUTE PROHIBITIONS

### NEVER SAY THESE PHRASES:
- "The backend is working" (when there are 500 errors)
- "It's mostly working" (partial success doesn't count)
- "The API is functional" (when endpoints return errors)
- "Charts are operational" (when they show no data)
- "Implementation is complete" (when core functionality broken)
- "The system is functional" (when core features broken)
- "Everything is set up" (when data flow is broken)

### ALWAYS BE HONEST:
- "The backend starts but has 500 errors on endpoints X, Y, Z"
- "PostgreSQL connects but no market data is flowing"
- "Charts render but show empty data because IB Gateway isn't connected"
- "Implementation is partial - core functionality still broken"
- "Backend has connection failures and empty data responses"
- "System is not working - charts show no real market data"

## 🚫 BACKEND NOT WORKING INDICATORS
Backend is NOT working if ANY of these exist:
- 500 Internal Server Error on any endpoint
- IB Gateway connection failures
- Empty data responses (candles: [])
- PostgreSQL connection errors
- Constant retry loops in logs
- Frontend showing "No data available"
- Client ID conflicts preventing connections
- Timeouts on data requests
- Mock or placeholder data being used

**NOTE**: Exit code 137 with recent 200 OK responses = user terminated working backend, NOT failure

## ✅ BACKEND WORKING DEFINITION
Backend is only "working" when:
- [ ] Starts without errors
- [ ] All endpoints return 200 OK (no 500 errors)
- [ ] PostgreSQL connected and storing data
- [ ] IB Gateway connected (no client_id conflicts)
- [ ] Market data flowing in real-time
- [ ] Charts show actual data (not empty arrays)
- [ ] Zero connection failure warnings in logs
- [ ] No timeouts on data requests
- [ ] All timeframes return real market data

**POSITIVE INDICATORS OF WORKING BACKEND:**
- Recent 200 OK responses in logs
- API endpoints successfully serving requests
- Active request processing without errors

## PERFORMANCE REQUIREMENTS
- **APIs must respond in milliseconds, never seconds**
- **Never wait more than 10s for any process**
- **If backend is unresponsive, restart immediately**
- **Dead processes should be killed and restarted, not waited for**

## ERROR HANDLING REQUIREMENTS
- Show clear error messages when backend fails
- Log actual API responses and errors
- Surface data availability problems to users
- Investigate root causes in backend/IB Gateway

## BACKEND SPECIFIC ISSUES

### Historical Data Timeouts
**Problem**: IB Gateway returning timeouts for longer timeframes
```
ERROR:root:Error getting historical bars for AAPL: Timeout waiting for historical data for AAPL
```

**Timeframes Affected**:
- 1M (Monthly)
- 1W (Weekly) 
- 1D (Daily)

**API Response**: Empty candles array `{"candles": []}`

**Required Investigation**:
1. Check IB Gateway historical data permissions
2. Verify contract definitions for longer timeframes
3. Investigate timeout configuration
4. Check data subscription status
5. Verify IB API version compatibility

### IB Gateway Warnings
```
Warning: Your API version does not support fractional share size rules. 
Please upgrade to a minimum version 163.
```

**Action Required**: Upgrade IB Gateway API version

### Contract Definition Errors
```
Error validating request.-'bM' : cause - Please enter a local symbol or an expiry
No security definition has been found for the request
```

**Action Required**: Fix contract symbol mapping for futures/commodities

### Current Known Issues

#### 1M Timeframe Problem
- **Issue**: Backend returns empty candles array for longer timeframes (1M, 1W, 1D)
- **Root Cause**: IB Gateway timeout/configuration issue
- **Status**: Backend investigation needed - NOT frontend mock data
- **Error**: "Timeout waiting for historical data"

#### Frontend Chart Status
- ✅ API calls correctly formatted with all parameters
- ✅ Timestamp parsing handles multiple formats
- ✅ Duplicate removal and sorting working
- ✅ Chart renders when data is available
- ❌ Backend data availability for longer timeframes

## 📋 PROJECT STATUS REALITY CHECK

### What Actually Works (verified):
- ✅ **IB Gateway Integration**: Connects with client ID management, no conflicts
- ✅ **Instrument Search**: PLTR search returns real stock data, not currency pairs
- ✅ **Frontend Asset Class Filtering**: Defaults to stocks only (sec_type=STK) for clean results
- ✅ **Backend Filtering Logic**: Properly filters by security type in ib_instrument_provider.py
- ✅ **Backfill Disconnect Handling**: Gracefully stops on IB Gateway disconnect, no hanging
- ✅ **API Endpoints**: `/api/v1/ib/instruments/search/{query}?sec_type=STK` returns real data
- ✅ **Connection Management**: Proper connect/disconnect cycles without resource leaks

### What's Actually Broken (be honest):
- ❌ **Historical Data Timeouts**: IB Gateway returns empty data for longer timeframes (1M, 1W, 1D)
- ❌ **Chart Data Availability**: Charts show empty arrays for monthly/weekly timeframes
- ❌ **IB API Version Warning**: API version 163+ required for fractional shares

### Required for "Working" Status:
- Backend runs with zero errors
- IB Gateway connected and streaming data
- PostgreSQL storing market data with nanosecond precision
- Charts showing real market data across all timeframes
- Parquet export generating NautilusTrader-compatible files
- Zero 500 errors, zero connection failures, zero empty data responses

## 🎯 PROJECT GOAL REMINDER
**Fix broken chart timeframes** - this means charts must show REAL MARKET DATA, not empty arrays.

If charts are not showing real market data across all timeframes, THE PROJECT GOAL IS NOT ACHIEVED.

---

**CLAUDE: Read this file first, follow these rules absolutely, use the checklist before any claims.**