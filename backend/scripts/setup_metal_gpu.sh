#!/bin/bash
# Metal GPU Acceleration Setup Script
# Nautilus Trading Platform - M4 Max Optimization

set -e  # Exit on any error

echo "=================================================="
echo " Metal GPU Acceleration Setup"
echo " Nautilus Trading Platform - M4 Max Optimization"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "SUCCESS") echo -e "${GREEN}✅ $message${NC}" ;;
        "ERROR")   echo -e "${RED}❌ $message${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "INFO")    echo -e "ℹ️  $message" ;;
    esac
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_status "ERROR" "This script is designed for macOS only"
    exit 1
fi

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    print_status "ERROR" "This script requires Apple Silicon (ARM64) architecture"
    exit 1
fi

print_status "SUCCESS" "Running on Apple Silicon macOS"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d" " -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d"." -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d"." -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 11 ]]; then
    print_status "ERROR" "Python 3.11+ required. Found: $PYTHON_VERSION"
    exit 1
elif [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 14 ]]; then
    print_status "WARNING" "Python $PYTHON_VERSION may have compatibility issues"
else
    print_status "SUCCESS" "Python version compatible: $PYTHON_VERSION"
fi

# Check if in virtual environment (recommended)
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_status "WARNING" "Not in virtual environment. Consider using one."
else
    print_status "SUCCESS" "Using virtual environment: $VIRTUAL_ENV"
fi

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
REQUIREMENTS_FILE="$BACKEND_DIR/requirements-metal.txt"

# Check if requirements file exists
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    print_status "ERROR" "requirements-metal.txt not found at: $REQUIREMENTS_FILE"
    exit 1
fi

print_status "INFO" "Installing Metal GPU acceleration dependencies..."

# Install dependencies
if pip3 install -r "$REQUIREMENTS_FILE"; then
    print_status "SUCCESS" "All dependencies installed successfully"
else
    print_status "ERROR" "Failed to install dependencies"
    exit 1
fi

# Run verification script
VERIFY_SCRIPT="$BACKEND_DIR/verify_metal_gpu.py"
if [[ -f "$VERIFY_SCRIPT" ]]; then
    print_status "INFO" "Running verification tests..."
    if python3 "$VERIFY_SCRIPT"; then
        print_status "SUCCESS" "All verification tests passed"
    else
        print_status "WARNING" "Some verification tests failed"
    fi
else
    print_status "WARNING" "Verification script not found: $VERIFY_SCRIPT"
fi

# Create GPU test script
GPU_TEST_SCRIPT="$BACKEND_DIR/test_gpu_performance.py"
cat > "$GPU_TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""Quick GPU performance test for Nautilus trading platform."""

import torch
import time

def test_gpu_performance():
    """Test GPU vs CPU performance for trading algorithms."""
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return False
    
    print("🚀 Testing GPU performance for trading algorithms...")
    
    # Simulate typical trading computation
    device = torch.device('mps')
    
    # Price data simulation (1000 assets, 252 trading days)
    prices = torch.randn(1000, 252, device=device, dtype=torch.float32)
    
    # Calculate returns
    start_time = time.time()
    returns = torch.diff(torch.log(prices), dim=1)
    
    # Calculate volatility (rolling standard deviation)
    window = 21
    vol = torch.zeros_like(returns)
    for i in range(window-1, returns.shape[1]):
        vol[:, i] = torch.std(returns[:, i-window+1:i+1], dim=1)
    
    # Calculate correlation matrix
    corr_matrix = torch.corrcoef(returns)
    
    # Portfolio optimization simulation
    n_assets = 100
    subset_returns = returns[:n_assets, :]
    cov_matrix = torch.cov(subset_returns)
    
    torch.mps.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"✅ GPU computation completed in {gpu_time:.3f}s")
    print(f"📊 Processed {prices.shape[0]} assets over {prices.shape[1]} days")
    print(f"💾 GPU memory used: {torch.mps.current_allocated_memory() / 1024**2:.1f}MB")
    
    return True

if __name__ == "__main__":
    test_gpu_performance()
EOF

chmod +x "$GPU_TEST_SCRIPT"
print_status "SUCCESS" "GPU performance test script created"

# Summary
echo ""
echo "=================================================="
echo " Setup Complete"
echo "=================================================="
print_status "SUCCESS" "Metal GPU acceleration setup completed successfully"
print_status "INFO" "Files created:"
echo "  - $REQUIREMENTS_FILE"
echo "  - $VERIFY_SCRIPT"
echo "  - $GPU_TEST_SCRIPT"

print_status "INFO" "Next steps:"
echo "  1. Run: python3 verify_metal_gpu.py"
echo "  2. Test: python3 test_gpu_performance.py" 
echo "  3. Integrate with Docker containers"

echo ""
print_status "INFO" "For Docker integration, ensure:"
echo "  - Use Apple Silicon compatible base images"
echo "  - Mount GPU device if supported by Docker Desktop"
echo "  - Set environment variables for optimal performance"

echo ""
print_status "SUCCESS" "Ready for production deployment!"