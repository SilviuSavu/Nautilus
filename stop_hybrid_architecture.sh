#!/bin/bash

# Nautilus Hybrid Architecture Shutdown Script
# Gracefully stops native engines and Docker infrastructure

set -e

echo "🛑 Shutting down Nautilus Hybrid Architecture"
echo "=============================================="

# Stop native engines
echo "🧠 Stopping native ML Engine..."
if [ -f /tmp/nautilus_ml.pid ]; then
    ML_PID=$(cat /tmp/nautilus_ml.pid)
    if kill -0 $ML_PID 2>/dev/null; then
        kill -TERM $ML_PID
        sleep 2
        if kill -0 $ML_PID 2>/dev/null; then
            kill -KILL $ML_PID
        fi
    fi
    rm -f /tmp/nautilus_ml.pid
fi

echo "🎲 Stopping native Risk Engine..."
if [ -f /tmp/nautilus_risk.pid ]; then
    RISK_PID=$(cat /tmp/nautilus_risk.pid)
    if kill -0 $RISK_PID 2>/dev/null; then
        kill -TERM $RISK_PID
        sleep 2
        if kill -0 $RISK_PID 2>/dev/null; then
            kill -KILL $RISK_PID
        fi
    fi
    rm -f /tmp/nautilus_risk.pid
fi

echo "📈 Stopping native Strategy Engine..."
if [ -f /tmp/nautilus_strategy.pid ]; then
    STRATEGY_PID=$(cat /tmp/nautilus_strategy.pid)
    if kill -0 $STRATEGY_PID 2>/dev/null; then
        kill -TERM $STRATEGY_PID
        sleep 2
        if kill -0 $STRATEGY_PID 2>/dev/null; then
            kill -KILL $STRATEGY_PID
        fi
    fi
    rm -f /tmp/nautilus_strategy.pid
fi

echo "📊 Stopping native Factor Engine..."
if [ -f /tmp/nautilus_factor.pid ]; then
    FACTOR_PID=$(cat /tmp/nautilus_factor.pid)
    if kill -0 $FACTOR_PID 2>/dev/null; then
        kill -TERM $FACTOR_PID
        sleep 2
        if kill -0 $FACTOR_PID 2>/dev/null; then
            kill -KILL $FACTOR_PID
        fi
    fi
    rm -f /tmp/nautilus_factor.pid
fi

echo "⚡ Stopping native Features Engine..."
if [ -f /tmp/nautilus_features.pid ]; then
    FEATURES_PID=$(cat /tmp/nautilus_features.pid)
    if kill -0 $FEATURES_PID 2>/dev/null; then
        kill -TERM $FEATURES_PID
        sleep 2
        if kill -0 $FEATURES_PID 2>/dev/null; then
            kill -KILL $FEATURES_PID
        fi
    fi
    rm -f /tmp/nautilus_features.pid
fi

# Kill any remaining native engine processes
pkill -f "native_ml_engine.py" 2>/dev/null || true
pkill -f "native_risk_engine.py" 2>/dev/null || true
pkill -f "native_strategy_engine.py" 2>/dev/null || true
pkill -f "native_factor_engine.py" 2>/dev/null || true
pkill -f "native_features_engine.py" 2>/dev/null || true

echo "🐳 Stopping Docker containers..."
docker-compose -f docker-compose.hybrid.yml down

# Clean up Unix sockets
echo "🧹 Cleaning up Unix sockets..."
rm -f /tmp/nautilus_*.sock
rm -f /tmp/nautilus_sockets/* 2>/dev/null || true

# Clean up shared memory
echo "🗑️  Cleaning up shared memory..."
rm -f /tmp/nautilus_shared_memory 2>/dev/null || true

echo "✅ Hybrid architecture shutdown complete"