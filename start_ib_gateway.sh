#!/bin/bash

# Interactive Brokers Gateway/TWS Setup Script
# This script provides instructions and utilities for starting IB Gateway or TWS

set -e

echo "=========================================="
echo "Interactive Brokers Gateway/TWS Setup"
echo "=========================================="

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS environment"
    
    # Check if TWS is already installed
    TWS_PATH="/Applications/Trader Workstation.app"
    GATEWAY_PATH="/Applications/IB Gateway.app"
    
    if [ -d "$TWS_PATH" ]; then
        echo "✓ TWS found at: $TWS_PATH"
        TWS_INSTALLED=true
    else
        echo "✗ TWS not found at expected location"
        TWS_INSTALLED=false
    fi
    
    if [ -d "$GATEWAY_PATH" ]; then
        echo "✓ IB Gateway found at: $GATEWAY_PATH" 
        GATEWAY_INSTALLED=true
    else
        echo "✗ IB Gateway not found at expected location"
        GATEWAY_INSTALLED=false
    fi
    
else
    echo "Non-macOS environment detected"
    TWS_INSTALLED=false
    GATEWAY_INSTALLED=false
fi

echo ""
echo "=========================================="
echo "Setup Instructions"
echo "=========================================="

if [ "$TWS_INSTALLED" = false ] && [ "$GATEWAY_INSTALLED" = false ]; then
    echo "STEP 1: Download and Install IB Software"
    echo "----------------------------------------"
    echo "Download from: https://www.interactivebrokers.com/en/trading/tws.php"
    echo ""
    echo "Options:"
    echo "1. TWS (Trader Workstation) - Full trading platform with charts and tools"
    echo "2. IB Gateway - Lightweight API-only gateway (recommended for automated trading)"
    echo ""
    echo "For this integration, IB Gateway is recommended."
    echo ""
fi

echo "STEP 2: Configure API Settings"
echo "------------------------------"
echo "1. Start TWS or IB Gateway"
echo "2. Log in with your IB account credentials"
echo "3. Go to 'File' -> 'Global Configuration' -> 'API' -> 'Settings'"
echo "4. Enable the following settings:"
echo "   ✓ Enable ActiveX and Socket Clients"
echo "   ✓ Socket port: 4002 (IB Gateway paper) or 4001 (live trading)"
echo "   ✓ Master API client ID: Leave blank or set to 0"
echo "   ✓ Read-Only API: Unchecked (to allow trading)"
echo "   ✓ Download open orders on connection: Checked"
echo "5. Click 'OK' and restart TWS/Gateway"
echo ""

echo "STEP 3: Configure Paper Trading (Recommended for Testing)"
echo "---------------------------------------------------------"
echo "1. In TWS/Gateway, go to 'File' -> 'Global Configuration' -> 'Paper Trading'"
echo "2. Enable 'Paper Trading Mode'"
echo "3. Note your paper trading account ID (usually starts with 'DU')"
echo "4. Update the account ID in .env file:"
echo "   IB_ACCOUNT_ID=DU12345  # Replace with your actual account ID"
echo ""

echo "STEP 4: Verify Connection Settings"
echo "----------------------------------"
echo "Current configuration in .env:"
echo "  Host: $(grep IB_HOST .env 2>/dev/null || echo '127.0.0.1')"
echo "  Port: $(grep IB_PORT .env 2>/dev/null || echo '7497')"
echo "  Client ID: $(grep IB_CLIENT_ID .env 2>/dev/null || echo '1')"
echo "  Account ID: $(grep IB_ACCOUNT_ID .env 2>/dev/null || echo 'DU12345')"
echo ""

echo "STEP 5: Start the Gateway/TWS"
echo "-----------------------------"
if [ "$TWS_INSTALLED" = true ]; then
    echo "To start TWS:"
    echo "  open '$TWS_PATH'"
fi

if [ "$GATEWAY_INSTALLED" = true ]; then
    echo "To start IB Gateway:"
    echo "  open '$GATEWAY_PATH'"
fi

if [ "$TWS_INSTALLED" = false ] && [ "$GATEWAY_INSTALLED" = false ]; then
    echo "Install TWS or IB Gateway first, then run this script again."
fi

echo ""
echo "STEP 6: Test Connection"
echo "----------------------"
echo "After starting TWS/Gateway with API enabled:"
echo "1. Run the NautilusTrader configuration test:"
echo "   python ib_config.py"
echo "2. Or run the backend integration test:"
echo "   cd backend && python test_ib_integration.py"
echo ""

echo "=========================================="
echo "Troubleshooting"
echo "=========================================="
echo "Common issues:"
echo "1. 'Connection refused' - Check if TWS/Gateway is running and API is enabled"
echo "2. 'Invalid client ID' - Make sure client ID is unique (use different IDs for multiple connections)"
echo "3. 'Authentication failed' - Check account credentials and paper trading mode"
echo "4. 'Market data not available' - Ensure you have market data subscriptions"
echo ""
echo "For more help, see the IB_INTEGRATION_README.md file."
echo ""

# Ask user if they want to open TWS/Gateway now
if [ "$TWS_INSTALLED" = true ] || [ "$GATEWAY_INSTALLED" = true ]; then
    echo "=========================================="
    read -p "Would you like to start IB Gateway/TWS now? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ "$GATEWAY_INSTALLED" = true ]; then
            echo "Starting IB Gateway..."
            open "$GATEWAY_PATH"
        elif [ "$TWS_INSTALLED" = true ]; then
            echo "Starting TWS..."
            open "$TWS_PATH"
        fi
        
        echo ""
        echo "Waiting for you to configure API settings..."
        echo "Once TWS/Gateway is configured and running with API enabled,"
        echo "press any key to continue with connection test..."
        read -n 1 -s
        
        echo ""
        echo "Testing connection..."
        if [ -f "ib_config.py" ]; then
            python ib_config.py
        else
            echo "ib_config.py not found. Run this script from the project root directory."
        fi
    fi
fi

echo ""
echo "Setup script completed!"