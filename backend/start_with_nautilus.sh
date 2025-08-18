#!/bin/bash

# Start backend with NautilusTrader and Python 3.13
# This script ensures we use the correct Python version with NautilusTrader

echo "🚀 Starting Nautilus Trader Backend with Python 3.13..."

# Setup pyenv environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# Verify Python version
PYTHON_VERSION=$(python --version)
echo "✓ Using Python: $PYTHON_VERSION"

# Verify NautilusTrader installation
python -c "
import nautilus_trader
from nautilus_trader.adapters.interactive_brokers.factories import InteractiveBrokersLiveDataClientFactory
print('✓ NautilusTrader IB adapter verified')
" || {
    echo "❌ NautilusTrader not properly installed"
    echo "Run: pip install 'nautilus_trader[ib]'"
    exit 1
}

# Start the backend
echo "🔄 Starting FastAPI backend on port 8000..."
cd "$(dirname "$0")"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

echo "🛑 Backend stopped"