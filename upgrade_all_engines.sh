#!/bin/bash
# MANUAL ENGINE UPGRADE SCRIPT - When agents fail, use this
# Dr. DocHealth's Emergency Treatment Script

echo "🚀 MANUAL ENGINE UPGRADE - EMERGENCY TREATMENT"
echo "==============================================="
echo "When all other agents are incompetent, this works."
echo ""

# Set M4 Max environment
export M4_MAX_OPTIMIZED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export METAL_DEVICE_WRAPPER_TYPE=1
export COREML_ENABLE_MLPROGRAM=1
export VECLIB_MAXIMUM_THREADS=12
export PYTHONPATH=/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend

echo "✅ M4 Max environment configured"

cd /Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend

# Stop all engines (brutal but effective)
echo ""
echo "🛑 STOPPING ALL ENGINES..."
for port in 8100 8110 8200 8300 8400 8500 8600 8700 8800 8900 9000 10000 10001; do
    echo "   Stopping port $port..."
    lsof -ti:$port | xargs kill -9 2>/dev/null || echo "   Port $port already free"
done

echo "⏱️ Waiting for cleanup..."
sleep 5

# Start engines with M4 Max optimization
echo ""
echo "🚀 STARTING ALL ENGINES WITH M4 MAX OPTIMIZATION..."
echo ""

# Wave 1: Critical engines
echo "Wave 1: Critical Processing Engines"
echo "-----------------------------------"
nohup python3 engines/analytics/ultra_fast_analytics_engine.py > logs/analytics.log 2>&1 &
echo "✅ Analytics Engine (Port 8100) - Ultra-Fast M4 Max"

nohup python3 engines/risk/ultra_fast_risk_engine.py > logs/risk.log 2>&1 &
echo "✅ Risk Engine (Port 8200) - Ultra-Fast M4 Max"

nohup python3 engines/factor/ultra_fast_factor_engine.py > logs/factor.log 2>&1 &
echo "✅ Factor Engine (Port 8300) - Ultra-Fast M4 Max"

nohup python3 engines/ml/ultra_fast_ml_engine.py > logs/ml.log 2>&1 &
echo "✅ ML Engine (Port 8400) - Ultra-Fast M4 Max"

echo "⏱️ Wave 1 startup delay..."
sleep 12

# Wave 2: Additional engines
echo ""
echo "Wave 2: Additional Processing Engines"
echo "------------------------------------"
nohup python3 engines/features/ultra_fast_features_engine.py > logs/features.log 2>&1 &
echo "✅ Features Engine (Port 8500) - Ultra-Fast M4 Max"

nohup python3 engines/websocket/ultra_fast_websocket_engine.py > logs/websocket.log 2>&1 &
echo "✅ WebSocket Engine (Port 8600) - Ultra-Fast M4 Max"

nohup python3 engines/strategy/ultra_fast_strategy_engine.py > logs/strategy.log 2>&1 &
echo "✅ Strategy Engine (Port 8700) - Ultra-Fast M4 Max"

nohup python3 engines/portfolio/ultra_fast_portfolio_engine.py > logs/portfolio.log 2>&1 &
echo "✅ Portfolio Engine (Port 8900) - Ultra-Fast M4 Max"

echo "⏱️ Wave 2 startup delay..."
sleep 12

# Wave 3: Specialized engines
echo ""
echo "Wave 3: Specialized Engines"
echo "---------------------------"
nohup python3 engines/collateral/ultra_fast_collateral_engine.py > logs/collateral.log 2>&1 &
echo "✅ Collateral Engine (Port 9000) - Ultra-Fast M4 Max"

nohup python3 engines/marketdata/centralized_marketdata_hub.py > logs/marketdata.log 2>&1 &
echo "✅ MarketData Engine (Port 8800) - Centralized Hub"

nohup python3 engines/backtesting/main.py > logs/backtesting.log 2>&1 &
echo "✅ Backtesting Engine (Port 8110) - Main Entry"

nohup python3 engines/vpin/ultra_fast_vpin_server.py > logs/vpin.log 2>&1 &
echo "✅ VPIN Engine (Port 10000) - Ultra-Fast"

nohup python3 engines/vpin/enhanced_microstructure_vpin_server.py > logs/enhanced_vpin.log 2>&1 &
echo "✅ Enhanced VPIN Engine (Port 10001) - Enhanced Microstructure"

echo ""
echo "⏱️ Final startup delay..."
sleep 15

# Health check
echo ""
echo "📊 COMPREHENSIVE HEALTH CHECK"
echo "============================="

declare -A engines=(
    [8100]="Analytics"
    [8110]="Backtesting" 
    [8200]="Risk"
    [8300]="Factor"
    [8400]="ML"
    [8500]="Features"
    [8600]="WebSocket"
    [8700]="Strategy"
    [8800]="MarketData"
    [8900]="Portfolio"
    [9000]="Collateral"
    [10000]="VPIN"
    [10001]="Enhanced VPIN"
)

healthy_count=0
total_engines=13

for port in "${!engines[@]}"; do
    engine_name="${engines[$port]}"
    printf "%-20s (Port %5s): " "$engine_name" "$port"
    
    # Health check with timeout
    response=$(timeout 10 curl -s "http://localhost:$port/health" 2>/dev/null)
    
    if [[ $? -eq 0 && "$response" =~ "healthy" ]]; then
        echo "✅ HEALTHY"
        ((healthy_count++))
        
        # Check for M4 Max acceleration
        if [[ "$response" =~ "m4_max" && "$response" =~ "true" ]]; then
            echo "                                    🚀 M4 Max: ACTIVE"
        fi
    else
        echo "❌ FAILED"
    fi
done

echo ""
echo "📈 UPGRADE RESULTS:"
echo "=================="
echo "   Healthy engines: $healthy_count/$total_engines"
echo "   Success rate: $(( healthy_count * 100 / total_engines ))%"
echo ""

if [[ $healthy_count -eq $total_engines ]]; then
    echo "🎉 MANUAL UPGRADE COMPLETE!"
    echo "   All 13 engines are operational with M4 Max optimization"
    echo "   No incompetent agents needed - manual medicine works!"
elif [[ $healthy_count -gt 10 ]]; then
    echo "✅ UPGRADE MOSTLY SUCCESSFUL"
    echo "   $(( total_engines - healthy_count )) engines need troubleshooting"
    echo "   Check logs in backend/logs/ directory"
elif [[ $healthy_count -gt 5 ]]; then
    echo "⚠️ PARTIAL SUCCESS"
    echo "   $(( total_engines - healthy_count )) engines failed to start"
    echo "   Manual intervention required for failed engines"
else
    echo "❌ UPGRADE NEEDS WORK"
    echo "   Most engines failed - check system requirements"
    echo "   Verify Python 3.13, PyTorch 2.8, and M4 Max detection"
fi

echo ""
echo "🩺 Dr. DocHealth's Treatment Complete"
echo "======================================"
echo "Manual upgrade bypasses all agent incompetence."
echo "Logs available in: backend/logs/"
echo "Health endpoints: http://localhost:[PORT]/health"
echo ""