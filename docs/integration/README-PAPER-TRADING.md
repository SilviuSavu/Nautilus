# 📄 Paper Trading with NautilusTrader

**Safe trading with REAL exchange APIs using FAKE money!**

## 🎯 Quick Start - Get Trading in 5 Minutes

### 1. Choose Your Exchange (Binance Testnet Recommended)

**Binance Testnet** - Easiest to set up:
- Visit: https://testnet.binance.vision/
- Register and verify email (no KYC required)
- Get free testnet BTC and USDT instantly

**Bybit Testnet** - Alternative option:
- Visit: https://testnet.bybit.com/
- Register and get automatic testnet USDT

### 2. Create API Keys

**For Binance Testnet:**
1. Login to https://testnet.binance.vision/
2. Go to **Account → API Management**
3. Click **Create API**
4. Enable **"Enable Trading"** permission
5. **Optional but recommended**: Restrict IP to your server
6. Copy your **API Key** and **Secret Key**

### 3. Configure NautilusTrader

**Option A: Environment Variables (Quick)**
```bash
export BINANCE_API_KEY="your_testnet_api_key"
export BINANCE_API_SECRET="your_testnet_secret"
export BINANCE_SANDBOX="true"
export BINANCE_TRADING_MODE="testnet"

# Start NautilusTrader
docker-compose up -d
```

**Option B: .env File (Recommended)**
```bash
# Copy the example file
cp .env.paper.example .env.paper

# Edit .env.paper with your testnet credentials
nano .env.paper

# Start with paper trading configuration
docker-compose --env-file .env.paper up -d
```

### 4. Get Free Testnet Money

**Binance Testnet:**
1. Login to https://testnet.binance.vision/
2. Go to **Wallet → Faucet**
3. Get free testnet BTC (~$50k fake value)
4. Get free testnet USDT (10,000 fake USDT)

### 5. Connect and Start Trading

```bash
# Get access token
TOKEN=$(curl -s -X POST -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' \
  http://localhost:8000/api/v1/auth/login | jq -r '.access_token')

# Connect to Binance testnet
curl -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/exchanges/binance/connect

# Check your fake money balances
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/portfolio/main/balances
```

## 🌐 Access Your Trading Dashboard

- **Main Dashboard**: http://localhost:3000/dashboard.html
- **Simple Dashboard**: http://localhost:3000/simple-dashboard.html  
- **React App**: http://localhost:3000 (may need browser refresh)
- **API Documentation**: http://localhost:8000/docs
- **Paper Trading Guide**: http://localhost:8000/api/v1/trading/paper-setup

## ✅ What You Get with Paper Trading

### Real Exchange Features
- **Real APIs**: Same API calls as live trading
- **Real Market Data**: Live prices and order books
- **Real Order Types**: Market, limit, stop orders
- **Real Latency**: Experience actual response times
- **Real Rate Limits**: Learn exchange limitations

### Safe Environment
- **Zero Risk**: Uses fake money (testnet tokens)
- **No KYC**: Most testnets don't require verification  
- **Free Funds**: Unlimited fake money for testing
- **Reset Anytime**: Clear and restart whenever needed
- **Learn Safely**: Make mistakes without financial loss

### Full Trading Experience
- **Portfolio Tracking**: Monitor fake positions and P&L
- **Risk Management**: Test stop losses and position sizing
- **Strategy Testing**: Validate algorithms before going live
- **Exchange Learning**: Understand each exchange's quirks
- **API Familiarity**: Learn endpoints and error handling

## 📊 Example Paper Trading Session

```bash
# 1. Check system status
curl http://localhost:8000/api/v1/status
# Response: {"trading_mode": "testnet", "features": {"trading": true}}

# 2. View exchange status  
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/exchanges/status
# Response: Shows Binance connected in testnet mode

# 3. Check your fake balances
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/portfolio/main/balances
# Response: Shows testnet BTC and USDT balances

# 4. Place a test order (fake money!)
# This would be done through the trading interface or API

# 5. Monitor your fake portfolio
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/portfolio/main/summary
# Response: Shows fake P&L and positions
```

## 🔧 Configuration Details

### Environment Variables for Testnet

```bash
# Binance Testnet (Recommended)
BINANCE_API_KEY=your_testnet_api_key
BINANCE_API_SECRET=your_testnet_secret  
BINANCE_SANDBOX=true
BINANCE_TRADING_MODE=testnet
BINANCE_BASE_URL=https://testnet.binance.vision

# Bybit Testnet
BYBIT_API_KEY=your_testnet_api_key
BYBIT_API_SECRET=your_testnet_secret
BYBIT_SANDBOX=true  
BYBIT_TRADING_MODE=testnet
BYBIT_BASE_URL=https://api-testnet.bybit.com

# Coinbase Sandbox
COINBASE_API_KEY=your_sandbox_api_key
COINBASE_API_SECRET=your_sandbox_secret
COINBASE_PASSPHRASE=your_sandbox_passphrase
COINBASE_SANDBOX=true
COINBASE_TRADING_MODE=testnet
COINBASE_BASE_URL=https://api-public.sandbox.exchange.coinbase.com
```

### Default Configuration (Safe)

NautilusTrader defaults to the safest settings:
- **SANDBOX=true** - Always use testnet/sandbox
- **TRADING_MODE=testnet** - Never accidentally go live
- **Testnet URLs** - Point to sandbox environments by default

## 🛡️ Safety Features

### Multiple Safety Layers
1. **Environment Separation**: Testnet configs separate from live
2. **Default Safety**: All exchanges default to sandbox mode
3. **Visual Indicators**: UI shows "PAPER" or "TESTNET" mode clearly  
4. **URL Validation**: Testnet URLs prevent live trading accidents
5. **Fake Money Only**: Impossible to lose real money

### Best Practices
- **Never mix environments**: Keep testnet and live configs separate
- **Test thoroughly**: Practice extensively before considering live trading
- **Document results**: Track what works and what doesn't
- **Start small**: Even in live trading, begin with tiny amounts
- **Have fun**: Experiment freely with no financial pressure!

## 🚀 Ready for Live Trading?

When you're confident with your paper trading results:

1. **Proven Strategy**: Consistently profitable on testnet
2. **Risk Management**: Tested stop losses and position sizing
3. **Technical Skills**: Comfortable with APIs and system operation
4. **Small Start**: Begin live trading with tiny amounts
5. **Gradual Scale**: Increase position sizes very slowly

## 📚 Additional Resources

- **Full Setup Guide**: `/docs/PAPER-TRADING-SETUP.md`
- **Trading Guide**: `/docs/TRADING-SETUP.md`  
- **API Documentation**: `http://localhost:8000/docs`
- **Exchange Testnets**:
  - Binance: https://testnet.binance.vision/
  - Bybit: https://testnet.bybit.com/
  - Coinbase: https://public.sandbox.exchange.coinbase.com

---

## 🎉 Start Paper Trading Now!

You're all set to begin safe paper trading with real exchange APIs and fake money. This is the best way to learn trading without risking capital.

**Remember**: Paper trading profits aren't real, but the learning is invaluable! 📈