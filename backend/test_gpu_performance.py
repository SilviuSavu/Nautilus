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
