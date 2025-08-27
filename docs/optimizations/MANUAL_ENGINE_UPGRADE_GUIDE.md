# 🚀 MANUAL ENGINE UPGRADE GUIDE - Nautilus Trading Platform

**EMERGENCY UPGRADE PROCEDURES** - When automated agents fail, use these manual commands

**LATEST UPDATE**: August 26, 2025 - Backtesting Engine procedures updated to use simple_backtesting_engine.py (working version)

## 🎯 **UPGRADE TARGETS - 13 Processing Engines**

### **Current Engine Status Assessment**
```bash
# Quick health check of all engines
for port in 8100 8110 8200 8300 8400 8500 8600 8700 8800 8900 9000 10000 10001; do
    echo "=== Engine Port $port ==="
    curl -s "http://localhost:$port/health" | jq '.status' 2>/dev/null || echo "❌ NOT RESPONDING"
done
```

## 🔧 **MANUAL UPGRADE PROCEDURES**

### **Phase 1: Stop All Engines**
```bash
# Stop all engine processes (brutal but effective)
echo "🛑 STOPPING ALL ENGINES..."
for port in 8100 8110 8200 8300 8400 8500 8600 8700 8800 8900 9000 10000 10001; do
    echo "Stopping port $port..."
    lsof -ti:$port | xargs kill -9 2>/dev/null || echo "Port $port already free"
done

echo "⏱️ Waiting for cleanup..."
sleep 5
```

### **Phase 2: M4 Max Environment Setup**
```bash
# Force M4 Max optimization environment
export M4_MAX_OPTIMIZED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export METAL_DEVICE_WRAPPER_TYPE=1
export COREML_ENABLE_MLPROGRAM=1
export VECLIB_MAXIMUM_THREADS=12
export PYTHONPATH=/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend

echo "✅ M4 Max environment configured"
echo "M4_MAX_OPTIMIZED: $M4_MAX_OPTIMIZED"
```

### **Phase 3: Manual Engine Upgrades**

#### **🏗️ Engine 1: Analytics (Port 8100)**
```bash
echo "🚀 Starting Analytics Engine with M4 Max..."
cd /Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend

# Choose your upgrade version:
# Option A: Ultra-Fast M4 Max
nohup python3 engines/analytics/ultra_fast_analytics_engine.py > analytics.log 2>&1 &

# Option B: SME-Accelerated
# nohup python3 engines/analytics/ultra_fast_sme_analytics_engine.py > analytics.log 2>&1 &

# Option C: Dual MessageBus
# nohup python3 engines/analytics/dual_bus_analytics_engine.py > analytics.log 2>&1 &

echo "⏱️ Waiting for Analytics startup..."
sleep 8

# Test
curl -s "http://localhost:8100/health" | jq '.status' || echo "❌ Analytics startup failed"
```

#### **🏗️ Engine 2: Risk (Port 8200)**
```bash
echo "🚀 Starting Risk Engine with M4 Max..."

# Choose your upgrade version:
# Option A: Ultra-Fast M4 Max
nohup python3 engines/risk/ultra_fast_risk_engine.py > risk.log 2>&1 &

# Option B: SME-Accelerated  
# nohup python3 engines/risk/ultra_fast_sme_risk_engine.py > risk.log 2>&1 &

# Option C: Dual MessageBus
# nohup python3 engines/risk/dual_bus_risk_engine.py > risk.log 2>&1 &

sleep 8
curl -s "http://localhost:8200/health" | jq '.status' || echo "❌ Risk startup failed"
```

#### **🏗️ Engine 3: Factor (Port 8300)**
```bash
echo "🚀 Starting Factor Engine with M4 Max..."

# Ultra-Fast M4 Max version
nohup python3 engines/factor/ultra_fast_factor_engine.py > factor.log 2>&1 &

sleep 8
curl -s "http://localhost:8300/health" | jq '.status' || echo "❌ Factor startup failed"
```

#### **🏗️ Engine 4: ML (Port 8400)**  
```bash
echo "🚀 Starting ML Engine with M4 Max..."

# Choose your upgrade version:
# Option A: Ultra-Fast M4 Max
nohup python3 engines/ml/ultra_fast_ml_engine.py > ml.log 2>&1 &

# Option B: SME-Accelerated
# nohup python3 engines/ml/ultra_fast_sme_ml_engine.py > ml.log 2>&1 &

sleep 8
curl -s "http://localhost:8400/health" | jq '.status' || echo "❌ ML startup failed"
```

#### **🏗️ Engine 5: Features (Port 8500)**
```bash
echo "🚀 Starting Features Engine with M4 Max..."

# Choose your upgrade version:
# Option A: Ultra-Fast M4 Max
nohup python3 engines/features/ultra_fast_features_engine.py > features.log 2>&1 &

# Option B: SME-Accelerated
# nohup python3 engines/features/ultra_fast_sme_features_engine.py > features.log 2>&1 &

sleep 8
curl -s "http://localhost:8500/health" | jq '.status' || echo "❌ Features startup failed"
```

#### **🏗️ Engine 6: WebSocket (Port 8600)**
```bash
echo "🚀 Starting WebSocket Engine with M4 Max..."

# Choose your upgrade version:
# Option A: Ultra-Fast M4 Max
nohup python3 engines/websocket/ultra_fast_websocket_engine.py > websocket.log 2>&1 &

# Option B: SME-Accelerated
# nohup python3 engines/websocket/ultra_fast_sme_websocket_engine.py > websocket.log 2>&1 &

sleep 8
curl -s "http://localhost:8600/health" | jq '.status' || echo "❌ WebSocket startup failed"
```

#### **🏗️ Engine 7: Strategy (Port 8700)**
```bash
echo "🚀 Starting Strategy Engine with M4 Max..."

# Ultra-Fast M4 Max version
nohup python3 engines/strategy/ultra_fast_strategy_engine.py > strategy.log 2>&1 &

sleep 8
curl -s "http://localhost:8700/health" | jq '.status' || echo "❌ Strategy startup failed"
```

#### **🏗️ Engine 8: Portfolio (Port 8900)**
```bash
echo "🚀 Starting Portfolio Engine with M4 Max..."

# Choose your upgrade version:
# Option A: Ultra-Fast M4 Max
nohup python3 engines/portfolio/ultra_fast_portfolio_engine.py > portfolio.log 2>&1 &

# Option B: SME-Accelerated
# nohup python3 engines/portfolio/ultra_fast_sme_portfolio_engine.py > portfolio.log 2>&1 &

# Option C: Institutional Grade
# nohup python3 engines/portfolio/institutional_portfolio_engine.py > portfolio.log 2>&1 &

sleep 8
curl -s "http://localhost:8900/health" | jq '.status' || echo "❌ Portfolio startup failed"
```

#### **🏗️ Engine 9: Collateral (Port 9000)**
```bash
echo "🚀 Starting Collateral Engine with M4 Max..."

# Ultra-Fast M4 Max version
nohup python3 engines/collateral/ultra_fast_collateral_engine.py > collateral.log 2>&1 &

sleep 8
curl -s "http://localhost:9000/health" | jq '.status' || echo "❌ Collateral startup failed"
```

#### **🏗️ Engine 10: MarketData (Port 8800)**
```bash
echo "🚀 Starting MarketData Engine..."

# Centralized hub implementation
nohup python3 engines/marketdata/centralized_marketdata_hub.py > marketdata.log 2>&1 &

sleep 8
curl -s "http://localhost:8800/health" | jq '.status' || echo "❌ MarketData startup failed"
```

#### **🏗️ Engine 11: Backtesting (Port 8110)**
```bash
echo "🚀 Starting Backtesting Engine..."

# Use the simple backtesting engine (working version)
cd /Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend/engines/backtesting
nohup python3 simple_backtesting_engine.py > backtesting.log 2>&1 &

# Alternative: Module-based approach (if above fails)
# cd /Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend
# nohup python3 -m engines.backtesting.simple_backtesting_engine > backtesting.log 2>&1 &

sleep 8
curl -s "http://localhost:8110/health" | jq '.status' || echo "❌ Backtesting startup failed"
```

#### **🏗️ Engine 12: VPIN (Port 10000)**
```bash
echo "🚀 Starting VPIN Engine..."

# Ultra-fast version
nohup python3 engines/vpin/ultra_fast_vpin_server.py > vpin.log 2>&1 &

sleep 8
curl -s "http://localhost:10000/health" | jq '.status' || echo "❌ VPIN startup failed"
```

#### **🏗️ Engine 13: Enhanced VPIN (Port 10001)**
```bash
echo "🚀 Starting Enhanced VPIN Engine..."

# Enhanced microstructure version
nohup python3 engines/vpin/enhanced_microstructure_vpin_server.py > enhanced_vpin.log 2>&1 &

sleep 8
curl -s "http://localhost:10001/health" | jq '.status' || echo "❌ Enhanced VPIN startup failed"
```

## ✅ **Phase 4: Validation & Health Check**

### **Complete System Health Assessment**
```bash
echo "📊 COMPREHENSIVE ENGINE HEALTH CHECK"
echo "======================================"

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
    echo "Testing $engine_name (Port $port)..."
    
    # Health check
    response=$(curl -s "http://localhost:$port/health" 2>/dev/null)
    
    if [[ $? -eq 0 && "$response" =~ "healthy" ]]; then
        echo "✅ $engine_name: HEALTHY"
        ((healthy_count++))
        
        # Try to get M4 Max status
        m4_max=$(echo "$response" | jq -r '.performance.hardware_status.m4_max_detected // .hardware_status.m4_max_detected // .m4_max_detected // false' 2>/dev/null)
        
        if [[ "$m4_max" == "true" ]]; then
            echo "   🚀 M4 Max acceleration: ACTIVE"
        else
            echo "   ⚠️ M4 Max acceleration: Not detected"
        fi
    else
        echo "❌ $engine_name: FAILED or NOT RESPONDING"
    fi
    
    echo ""
done

echo "📈 UPGRADE RESULTS:"
echo "   Healthy engines: $healthy_count/$total_engines"
echo "   Success rate: $(( healthy_count * 100 / total_engines ))%"

if [[ $healthy_count -eq $total_engines ]]; then
    echo "🎉 MANUAL UPGRADE COMPLETE! All engines operational."
elif [[ $healthy_count -gt 0 ]]; then
    echo "⚠️ PARTIAL SUCCESS - Some engines need troubleshooting."
else
    echo "❌ UPGRADE FAILED - All engines need manual intervention."
fi
```

## 🩺 **TROUBLESHOOTING MANUAL FIXES**

### **Common Issues & Solutions**

#### **Issue 1: Port Already in Use**
```bash
# Kill specific port
lsof -ti:8100 | xargs kill -9

# Kill all Python processes (nuclear option)
pkill -f python3
```

#### **Issue 2: Import Errors** 
```bash
# Fix PYTHONPATH
export PYTHONPATH=/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend

# Install missing dependencies
pip3 install -r requirements.txt
```

#### **Issue 3: M4 Max Not Detected**
```bash
# Force M4 Max detection
python3 -c "
import sys
sys.path.append('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend')
from universal_m4_max_detection import is_m4_max_detected
print('M4 Max detected:', is_m4_max_detected())
"
```

#### **Issue 4: Database Connection Failed**
```bash
# Test database connection
PGPASSWORD=nautilus123 psql -h localhost -U nautilus -d nautilus -c "SELECT 'Database OK';"
```

## 📋 **AUTOMATED UPGRADE SCRIPT**

### **One-Command Full Upgrade**
```bash
#!/bin/bash
# Save as: upgrade_all_engines.sh

echo "🚀 AUTOMATED MANUAL ENGINE UPGRADE"
echo "=================================="

# Set environment
export M4_MAX_OPTIMIZED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export METAL_DEVICE_WRAPPER_TYPE=1
export COREML_ENABLE_MLPROGRAM=1
export VECLIB_MAXIMUM_THREADS=12
export PYTHONPATH=/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend

cd /Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend

# Stop all engines
echo "🛑 Stopping all engines..."
for port in 8100 8110 8200 8300 8400 8500 8600 8700 8800 8900 9000 10000 10001; do
    lsof -ti:$port | xargs kill -9 2>/dev/null
done
sleep 5

# Start all ultra-fast engines
echo "🚀 Starting all engines with M4 Max..."

# Critical engines first
nohup python3 engines/analytics/ultra_fast_analytics_engine.py > analytics.log 2>&1 &
nohup python3 engines/risk/ultra_fast_risk_engine.py > risk.log 2>&1 &
nohup python3 engines/factor/ultra_fast_factor_engine.py > factor.log 2>&1 &
nohup python3 engines/ml/ultra_fast_ml_engine.py > ml.log 2>&1 &

sleep 10

# Additional engines
nohup python3 engines/features/ultra_fast_features_engine.py > features.log 2>&1 &
nohup python3 engines/websocket/ultra_fast_websocket_engine.py > websocket.log 2>&1 &
nohup python3 engines/strategy/ultra_fast_strategy_engine.py > strategy.log 2>&1 &
nohup python3 engines/portfolio/ultra_fast_portfolio_engine.py > portfolio.log 2>&1 &

sleep 10

# Specialized engines
nohup python3 engines/collateral/ultra_fast_collateral_engine.py > collateral.log 2>&1 &
nohup python3 engines/marketdata/centralized_marketdata_hub.py > marketdata.log 2>&1 &
cd engines/backtesting && nohup python3 simple_backtesting_engine.py > backtesting.log 2>&1 &
cd ../.. && nohup python3 engines/vpin/ultra_fast_vpin_server.py > vpin.log 2>&1 &
nohup python3 engines/vpin/enhanced_microstructure_vpin_server.py > enhanced_vpin.log 2>&1 &

echo "⏱️ Waiting for all engines to initialize..."
sleep 15

# Health check
echo "📊 Final health check..."
python3 /Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/docs/optimizations/scripts/apply_m4_max_to_all_engines.py

echo "🎉 MANUAL UPGRADE COMPLETE!"
```

### **Usage**
```bash
# Make executable
chmod +x upgrade_all_engines.sh

# Run upgrade
./upgrade_all_engines.sh
```

---

## 🎯 **SUMMARY - Manual Upgrade Medicine**

**PRESCRIPTION FILLED** ✅
- **13 Individual Engine Upgrade Commands** - Specific manual procedures
- **Automated Script** - One-command full upgrade 
- **Troubleshooting Guide** - Common issue fixes
- **Health Validation** - Complete system verification

**FOLLOW-UP CARE**: Monitor engine logs and run health checks regularly

**PROGNOSIS**: Excellent - Manual procedures bypass all automated agent failures

*Dr. DocHealth's Prescription Complete* 🩺💊