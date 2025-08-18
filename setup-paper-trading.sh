#!/bin/bash

# NautilusTrader Paper Trading Setup Script
# This script helps you quickly configure paper trading with exchange testnets

set -e

echo "🎯 NautilusTrader Paper Trading Setup"
echo "========================================"
echo ""
echo "This script will help you set up SAFE paper trading using real exchange"
echo "testnets with FAKE money. Perfect for learning and testing strategies!"
echo ""

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Offer to create .env.paper file
if [ ! -f ".env.paper" ]; then
    echo "📝 Creating .env.paper configuration file..."
    
    if [ -f ".env.paper.example" ]; then
        cp .env.paper.example .env.paper
        echo "✅ Created .env.paper from example file"
    else
        # Create basic .env.paper file
        cat > .env.paper << EOF
# NautilusTrader Paper Trading Configuration
# Fill in your testnet API credentials below

# Binance Testnet (Recommended - get credentials at https://testnet.binance.vision/)
BINANCE_API_KEY=your_binance_testnet_api_key_here
BINANCE_API_SECRET=your_binance_testnet_secret_here
BINANCE_SANDBOX=true
BINANCE_TRADING_MODE=testnet
BINANCE_BASE_URL=https://testnet.binance.vision

# Bybit Testnet (Alternative - get credentials at https://testnet.bybit.com/)
BYBIT_API_KEY=your_bybit_testnet_api_key_here
BYBIT_API_SECRET=your_bybit_testnet_secret_here
BYBIT_SANDBOX=true
BYBIT_TRADING_MODE=testnet
BYBIT_BASE_URL=https://api-testnet.bybit.com

# General Settings
ENVIRONMENT=development
TRADING_MODE=testnet
EOF
        echo "✅ Created basic .env.paper file"
    fi
    echo ""
fi

# Check if credentials are configured
if grep -q "your_.*_api_key_here" .env.paper; then
    echo "⚠️  SETUP REQUIRED"
    echo ""
    echo "Your .env.paper file needs testnet API credentials."
    echo ""
    echo "QUICK START (Recommended):"
    echo "1. Go to: https://testnet.binance.vision/"
    echo "2. Register and verify email (no KYC required)"
    echo "3. Go to Account → API Management"
    echo "4. Create API key with 'Enable Trading' permission"
    echo "5. Edit .env.paper and replace the placeholder values:"
    echo "   - BINANCE_API_KEY=your_actual_testnet_key"
    echo "   - BINANCE_API_SECRET=your_actual_testnet_secret"
    echo ""
    echo "💰 Don't forget to get free testnet funds:"
    echo "   - Login to https://testnet.binance.vision/"
    echo "   - Go to Wallet → Faucet"
    echo "   - Get free testnet BTC and USDT (fake money!)"
    echo ""
    
    read -p "📝 Edit .env.paper now? (y/n): " edit_now
    if [ "$edit_now" = "y" ] || [ "$edit_now" = "Y" ]; then
        if command -v nano &> /dev/null; then
            nano .env.paper
        elif command -v vim &> /dev/null; then
            vim .env.paper
        else
            echo "Please edit .env.paper with your preferred editor"
        fi
    fi
    echo ""
fi

# Ask if user wants to start the system
echo "🚀 Ready to start NautilusTrader in paper trading mode!"
echo ""
read -p "Start the system now? (y/n): " start_now

if [ "$start_now" = "y" ] || [ "$start_now" = "Y" ]; then
    echo ""
    echo "🔄 Starting NautilusTrader with paper trading configuration..."
    echo ""
    
    # Stop any existing containers
    docker-compose down > /dev/null 2>&1 || true
    
    # Start with paper trading environment
    docker-compose --env-file .env.paper up -d
    
    echo ""
    echo "⏳ Waiting for services to start..."
    sleep 10
    
    # Check if services are running
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is running"
    else
        echo "⚠️ Backend may still be starting up"
    fi
    
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Frontend is running"
    else
        echo "⚠️ Frontend may still be starting up"
    fi
    
    echo ""
    echo "🎉 NautilusTrader Paper Trading is ready!"
    echo ""
    echo "📱 Access your trading dashboard:"
    echo "   • Main Dashboard: http://localhost:3000/dashboard.html"
    echo "   • Simple Dashboard: http://localhost:3000/simple-dashboard.html"
    echo "   • React App: http://localhost:3000 (may need refresh)"
    echo "   • API Docs: http://localhost:8000/docs"
    echo "   • Setup Guide: http://localhost:8000/api/v1/trading/paper-setup"
    echo ""
    echo "🔐 Default login:"
    echo "   • Username: admin"
    echo "   • Password: admin123"
    echo ""
    echo "🛡️ SAFETY REMINDER:"
    echo "   ✅ You're using TESTNET with FAKE money"
    echo "   ✅ Completely safe to experiment and learn"
    echo "   ✅ No real money at risk!"
    echo ""
    echo "📖 Next steps:"
    echo "   1. Login to the web interface"
    echo "   2. Check exchange connection status"
    echo "   3. View your testnet balances"
    echo "   4. Start paper trading!"
    echo ""
else
    echo ""
    echo "📋 To start manually later:"
    echo "   docker-compose --env-file .env.paper up -d"
    echo ""
fi

echo "📚 For detailed setup instructions, see:"
echo "   • README-PAPER-TRADING.md"
echo "   • docs/PAPER-TRADING-SETUP.md"
echo ""
echo "Happy paper trading! 📈"