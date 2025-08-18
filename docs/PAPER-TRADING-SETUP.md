# Paper Trading Setup Guide - Real Exchange Testnets

This guide shows how to set up **real paper trading** using actual exchange testnet/sandbox environments. This is far superior to simulated trading as you're using the real exchange APIs with fake money.

## 🎯 Why Real Paper Trading?

- **Real API behavior**: Same latency, rate limits, and responses as live trading
- **Real market data**: Live order books and market conditions
- **Real order management**: Actual order types, fills, and rejections
- **Risk-free testing**: Test strategies with fake money before going live
- **API familiarity**: Learn exchange-specific quirks and behavior

## 🏛️ Exchange Testnet Environments

### 1. Binance Testnet 🥇 **RECOMMENDED**

**Testnet URL**: https://testnet.binance.vision/

**Features**:
- Full spot trading simulation
- Real-time market data
- All order types supported
- Free testnet BTC/USDT

**Setup Steps**:
1. Visit https://testnet.binance.vision/
2. Click "Register" (no KYC required)
3. Create account and verify email
4. Go to API Management
5. Create API key with "Enable Trading" permissions
6. **Important**: Whitelist your server IP if needed

**Testnet Funding**:
- Get free testnet USDT: https://testnet.binance.vision/en/support/faq/115003814592
- Testnet BTC faucet available in your account

### 2. Bybit Testnet 🥈

**Testnet URL**: https://testnet.bybit.com/

**Features**:
- Derivatives and spot trading
- Real market conditions
- Free testnet funds

**Setup Steps**:
1. Visit https://testnet.bybit.com/
2. Register new account
3. Go to API Management
4. Create API key with trading permissions
5. Copy API key and secret

**Testnet Funding**:
- Automatic testnet USDT allocation
- Request more via support if needed

### 3. OKX Testnet

**Note**: OKX provides sandbox environment for institutional clients

### 4. Coinbase Advanced Trade Sandbox

**Sandbox URL**: https://public.sandbox.exchange.coinbase.com

**Features**:
- Full REST and WebSocket API
- Simulated order matching
- Real market data feeds

**Setup Steps**:
1. Create sandbox account at Coinbase Pro Sandbox
2. Generate sandbox API credentials
3. Use sandbox endpoints for all requests

### 5. Kraken Futures Testnet

**Note**: Kraken provides demo environments for futures trading

## ⚙️ NautilusTrader Configuration

### Environment Variables Setup

Create a `.env.paper` file for paper trading configuration:

```bash
# ===========================================
# PAPER TRADING CONFIGURATION - SAFE MODE
# ===========================================

# General Settings
TRADING_MODE=testnet
ENVIRONMENT=development

# Binance Testnet Configuration (RECOMMENDED)
BINANCE_API_KEY=your_binance_testnet_api_key
BINANCE_API_SECRET=your_binance_testnet_secret
BINANCE_SANDBOX=true
BINANCE_TRADING_MODE=testnet
BINANCE_BASE_URL=https://testnet.binance.vision
BINANCE_WS_URL=wss://testnet.binance.vision

# Bybit Testnet Configuration
BYBIT_API_KEY=your_bybit_testnet_api_key
BYBIT_API_SECRET=your_bybit_testnet_secret
BYBIT_SANDBOX=true
BYBIT_TRADING_MODE=testnet
BYBIT_BASE_URL=https://api-testnet.bybit.com

# Coinbase Sandbox Configuration
COINBASE_API_KEY=your_coinbase_sandbox_api_key
COINBASE_API_SECRET=your_coinbase_sandbox_secret
COINBASE_PASSPHRASE=your_coinbase_sandbox_passphrase
COINBASE_SANDBOX=true
COINBASE_TRADING_MODE=testnet
COINBASE_BASE_URL=https://api-public.sandbox.exchange.coinbase.com

# Risk Management for Paper Trading
MAX_POSITION_SIZE=10000  # $10k max position (fake money)
MAX_DAILY_TRADES=100     # Higher limits for testing
ENABLE_ALL_PAIRS=true    # Test with all available pairs
```

### Docker Compose for Paper Trading

Update your docker-compose to use the paper trading environment:

```bash
# Start with paper trading configuration
docker-compose --env-file .env.paper up -d
```

Or modify `docker-compose.yml` to default to testnet:

```yaml
environment:
  - TRADING_MODE=testnet
  - BINANCE_SANDBOX=true
  - BYBIT_SANDBOX=true
  - COINBASE_SANDBOX=true
```

## 🚀 Quick Setup Guide

### Step 1: Create Binance Testnet Account (Fastest)

1. **Register**: Go to https://testnet.binance.vision/
2. **Verify**: Check email and verify account
3. **API Key**: 
   - Go to Account → API Management
   - Click "Create API"
   - Enable "Enable Trading"
   - **Important**: Restrict IP to your server's IP
   - Save API Key and Secret

### Step 2: Configure NautilusTrader

Create `.env.paper` file:

```bash
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_secret_here
BINANCE_SANDBOX=true
BINANCE_TRADING_MODE=testnet
```

### Step 3: Start Paper Trading

```bash
# Use paper trading environment
docker-compose --env-file .env.paper up -d

# Or set environment variables
export BINANCE_API_KEY="your_testnet_key"
export BINANCE_API_SECRET="your_testnet_secret"
export BINANCE_SANDBOX="true"
export BINANCE_TRADING_MODE="testnet"

docker-compose up -d
```

### Step 4: Verify Connection

```bash
# Check exchange status
curl -H "Authorization: Bearer YOUR_JWT" \
     http://localhost:8000/api/v1/exchanges/status

# Should show Binance as "connected" with trading_mode: "testnet"
```

## 💰 Getting Testnet Funds

### Binance Testnet

1. **Login** to https://testnet.binance.vision/
2. **Wallet**: Go to Wallet → Faucet
3. **Request**: 
   - Get 1 testnet BTC (worth ~$50k fake)
   - Get 10,000 testnet USDT
   - Get testnet BNB for fees
4. **Refresh**: Funds appear instantly

### Bybit Testnet

1. **Auto-funding**: New accounts get automatic testnet USDT
2. **Request more**: Contact support if you need additional funds
3. **Reset**: Can reset account balance if needed

## 📊 Testing Strategy

### 1. Start Simple

```bash
# Test basic connectivity
curl -X POST -H "Authorization: Bearer TOKEN" \
     http://localhost:8000/api/v1/exchanges/binance/connect

# Check balances
curl -H "Authorization: Bearer TOKEN" \
     http://localhost:8000/api/v1/portfolio/main/balances
```

### 2. Place Test Orders

- Start with small amounts (0.01 BTC, 100 USDT)
- Test different order types (market, limit, stop)
- Verify order fills and balance updates

### 3. Strategy Testing

- Test your trading algorithms
- Verify risk management
- Monitor performance metrics

## 🔒 Security for Paper Trading

Even though it's fake money, follow security best practices:

### API Key Security

```bash
# Restrict permissions to trading only
# Set IP restrictions in exchange settings
# Use separate keys for testnet vs production
# Never commit keys to git repositories
```

### Environment Separation

```bash
# Keep testnet and live configurations separate
.env.paper     # Paper trading only
.env.live      # Never use in development!
```

## 🧪 Advanced Testing Features

### Multiple Exchange Testing

Test arbitrage and multi-exchange strategies:

```bash
# Configure multiple testnets
BINANCE_TRADING_MODE=testnet
BYBIT_TRADING_MODE=testnet
COINBASE_TRADING_MODE=testnet
```

### High-Frequency Testing

```bash
# Higher rate limits for testing
BINANCE_MAX_ORDERS_PER_MINUTE=1000
ENABLE_AGGRESSIVE_TESTING=true
```

## 📈 Monitoring Paper Trading

### Real-Time Dashboard

- **Portfolio**: Monitor fake P&L and positions
- **Orders**: Track order fills and rejections
- **Performance**: Analyze strategy performance
- **Risk**: Monitor risk metrics

### Logging and Analytics

```bash
# Enable detailed logging
LOG_LEVEL=DEBUG
ENABLE_TRADE_LOGGING=true

# Check logs
docker-compose logs backend | grep -i "trade\|order\|fill"
```

## ⚠️ Important Notes

### Testnet vs Live Differences

1. **Market Data**: Testnet may have slightly different prices
2. **Liquidity**: Lower liquidity in testnet order books
3. **Latency**: May be higher than production
4. **Features**: Some advanced features may not be available

### Best Practices

1. **Start Here**: Always test strategies on testnet first
2. **Real Conditions**: Use real market hours and conditions
3. **Risk Management**: Test your risk management thoroughly
4. **Performance**: Don't expect identical performance on live
5. **Documentation**: Document your testing results

### Migration to Live Trading

When ready for live trading:

1. **Proven Strategy**: Thoroughly tested on testnet
2. **Small Start**: Begin with tiny positions
3. **Monitor Closely**: Watch every trade initially
4. **Gradual Scale**: Increase position sizes slowly
5. **Stop Loss**: Always have exit strategies

## 🆘 Troubleshooting

### Common Issues

**"Invalid API Key"**
- Verify you're using testnet keys on testnet URLs
- Check API key permissions include trading

**"Insufficient Balance"**
- Request testnet funds from exchange faucet
- Verify balance endpoints are working

**"Order Rejected"**
- Check minimum order sizes
- Verify symbol names match testnet format
- Ensure sufficient balance for fees

### Getting Help

- **Binance Testnet**: https://testnet.binance.vision/en/support
- **Bybit Testnet**: https://testnet.bybit.com/app/help-center
- **NautilusTrader**: Check logs and API documentation

---

## 🎉 Ready to Paper Trade!

You're now set up to:
- ✅ Trade with real exchange APIs using fake money
- ✅ Test strategies safely before risking capital
- ✅ Learn exchange-specific behavior and quirks
- ✅ Build confidence in your trading system

**Remember**: Paper trading profits don't put food on the table, but paper trading losses don't hurt either! Use this environment to become a better trader. 📚