# Live Trading Configuration Guide

This guide explains how to configure NautilusTrader for live cryptocurrency trading with secure API credential management.

## ⚠️ IMPORTANT SECURITY NOTICE

**NEVER commit your actual API keys to version control!** Always use environment variables or secure credential management systems.

## Supported Exchanges

NautilusTrader supports live trading on the following exchanges:

- **Binance** (Spot & Futures)
- **Coinbase Advanced Trade**
- **Kraken** (Spot & Futures) 
- **Bybit** (Spot & Derivatives)
- **OKX** (Spot, Futures & Options)

## Trading Modes

1. **Paper Trading** (`paper`) - Simulated trading with fake money
2. **Testnet/Sandbox** (`testnet`) - Exchange testnet with fake money
3. **Live Trading** (`live`) - Real trading with real money ⚠️

## Environment Variables Setup

### Option 1: Using .env file (Recommended for Development)

Create a `.env.local` file in the project root (this file should NOT be committed):

```bash
# Binance Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
BINANCE_SANDBOX=true
BINANCE_TRADING_MODE=paper

# Coinbase Configuration
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here
COINBASE_SANDBOX=true
COINBASE_TRADING_MODE=paper

# Kraken Configuration
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_API_SECRET=your_kraken_api_secret_here

# Bybit Configuration
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_API_SECRET=your_bybit_api_secret_here

# OKX Configuration
OKX_API_KEY=your_okx_api_key_here
OKX_API_SECRET=your_okx_api_secret_here
OKX_PASSPHRASE=your_okx_passphrase_here
```

### Option 2: Using System Environment Variables

Export environment variables in your shell:

```bash
export BINANCE_API_KEY="your_binance_api_key_here"
export BINANCE_API_SECRET="your_binance_api_secret_here"
export BINANCE_TRADING_MODE="paper"
```

### Option 3: Using Docker Environment File

Create a `.env.local` file and use it with docker-compose:

```bash
docker-compose --env-file .env.local up
```

## Exchange-Specific Setup Instructions

### Binance

1. Create API key at: https://www.binance.com/en/my/settings/api-management
2. Required permissions: "Enable Reading", "Enable Spot & Margin Trading"
3. For testnet: https://testnet.binance.vision/

### Coinbase Advanced Trade

1. Create API key at: https://www.coinbase.com/settings/api
2. Required permissions: "View", "Trade"
3. For sandbox: https://docs.cloud.coinbase.com/advanced-trade-api/docs/ws-auth

### Kraken

1. Create API key at: https://www.kraken.com/u/settings/api
2. Required permissions: "Query Funds", "Query Open Orders & Trades", "Create & Modify Orders"

### Bybit

1. Create API key at: https://www.bybit.com/app/user/api-management
2. Required permissions: "Read-Write"
3. For testnet: https://testnet.bybit.com/

### OKX

1. Create API key at: https://www.okx.com/account/my-api
2. Required permissions: "Trade"
3. For testnet: Use sandbox environment

## Security Best Practices

### API Key Security

1. **Restrict IP Addresses**: Only allow API access from your server's IP
2. **Minimal Permissions**: Only grant necessary permissions
3. **Separate Keys**: Use different keys for different purposes
4. **Regular Rotation**: Rotate API keys periodically
5. **Monitor Usage**: Regularly check API key usage logs

### Environment Variable Security

1. **Never Commit**: Add `.env.local` to `.gitignore`
2. **File Permissions**: Set restrictive permissions (`chmod 600 .env.local`)
3. **Secure Storage**: Use secure credential managers in production
4. **Encryption**: Encrypt credential files at rest

### Trading Security

1. **Start Small**: Begin with small position sizes
2. **Test Thoroughly**: Use paper trading and testnet first
3. **Set Limits**: Configure position size and risk limits
4. **Monitor Actively**: Watch trading activity closely
5. **Emergency Stops**: Know how to quickly halt trading

## Configuration Examples

### Paper Trading Configuration

```bash
# Safe for testing - no real money involved
BINANCE_TRADING_MODE=paper
BINANCE_SANDBOX=true
BINANCE_API_KEY=demo_key
BINANCE_API_SECRET=demo_secret
```

### Testnet Configuration

```bash
# Uses exchange testnet - fake money but real API
BINANCE_TRADING_MODE=testnet
BINANCE_SANDBOX=true
BINANCE_API_KEY=your_testnet_api_key
BINANCE_API_SECRET=your_testnet_api_secret
```

### Live Trading Configuration ⚠️

```bash
# REAL MONEY TRADING - BE CAREFUL!
BINANCE_TRADING_MODE=live
BINANCE_SANDBOX=false
BINANCE_API_KEY=your_live_api_key
BINANCE_API_SECRET=your_live_api_secret
```

## Testing Your Configuration

1. **Start the services:**
   ```bash
   docker-compose up
   ```

2. **Check exchange status:**
   ```bash
   curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
        http://localhost:8000/api/v1/exchanges/status
   ```

3. **Test connection to specific exchange:**
   ```bash
   curl -X POST \
        -H "Authorization: Bearer YOUR_JWT_TOKEN" \
        http://localhost:8000/api/v1/exchanges/binance/connect
   ```

## Risk Management

### Position Limits

Configure maximum position sizes to limit risk:

```bash
BINANCE_POSITION_LIMIT=1000  # Max $1000 per position
BINANCE_MAX_ORDERS_PER_MINUTE=10  # Rate limiting
```

### Portfolio Limits

The system enforces these default risk limits:

- Maximum portfolio risk per trade: 2%
- Maximum position size: 10% of portfolio
- Maximum daily loss: 5%
- Maximum leverage: 3x

## Troubleshooting

### Common Issues

1. **"Invalid API Key"**: Check key format and permissions
2. **"IP Not Whitelisted"**: Add your server IP to exchange whitelist
3. **"Insufficient Permissions"**: Grant required permissions to API key
4. **"Rate Limited"**: Reduce API call frequency
5. **"Connection Failed"**: Check network connectivity and firewall

### Debugging

Enable debug logging to troubleshoot connection issues:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

### Log Analysis

Check container logs for detailed error messages:

```bash
docker-compose logs backend | grep -i "exchange\|trading\|error"
```

## Production Deployment

### Secure Credential Management

For production deployments, use:

1. **AWS Secrets Manager**
2. **HashiCorp Vault**
3. **Kubernetes Secrets**
4. **Azure Key Vault**
5. **Google Secret Manager**

### Monitoring

Set up monitoring for:

1. **Connection Status**: Exchange connectivity
2. **Trading Activity**: Order fills and rejections
3. **Risk Metrics**: Position sizes and P&L
4. **API Usage**: Rate limiting and quotas
5. **Security Events**: Failed authentication attempts

## Support

- **Documentation**: Check `/docs` endpoint for API documentation
- **Logs**: Review container logs for error details
- **Status**: Monitor `/api/v1/status` for system health
- **Community**: NautilusTrader Discord and GitHub

---

**Remember**: Always start with paper trading and thoroughly test your configuration before risking real money!