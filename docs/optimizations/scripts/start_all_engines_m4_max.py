#!/usr/bin/env python3
"""
Start all 13 engines with M4 Max optimization enabled
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable M4 Max optimization globally
os.environ['M4_MAX_OPTIMIZED'] = '1'
os.environ['PYTHONPATH'] = str(backend_path)

# Engine configurations: (name, port, script_path)
ENGINES = [
    ("Analytics", 8100, "engines/analytics/ultra_fast_analytics_engine.py"),
    ("Backtesting", 8110, "engines/backtesting/simple_backtesting_engine.py"), 
    ("Risk", 8200, "engines/risk/ultra_fast_risk_engine.py"),
    ("Factor", 8300, "engines/factor/ultra_fast_factor_engine.py"),
    ("ML", 8400, "engines/ml/ultra_fast_ml_engine.py"),
    ("Features", 8500, "engines/features/ultra_fast_features_engine.py"),
    ("WebSocket", 8600, "engines/websocket/ultra_fast_websocket_engine.py"),
    ("Strategy", 8700, "engines/strategy/ultra_fast_strategy_engine.py"),
    ("MarketData", 8800, "engines/marketdata/simple_marketdata_engine.py"),
    ("Portfolio", 8900, "engines/portfolio/ultra_fast_portfolio_engine.py"),
    ("Collateral", 9000, "engines/collateral/ultra_fast_collateral_engine.py"),
    ("VPIN", 10000, "engines/vpin/ultra_fast_vpin_server.py"),
    ("Enhanced VPIN", 10001, "engines/vpin/enhanced_microstructure_vpin_server.py"),
]

def start_engine(name: str, port: int, script_path: str) -> bool:
    """Start a single engine with M4 Max optimization"""
    try:
        full_path = backend_path / script_path
        
        if not full_path.exists():
            # Try alternative naming patterns
            alternatives = [
                script_path.replace("ultra_fast_", "simple_"),
                script_path.replace("ultra_fast_", ""),
                script_path.replace("_engine.py", ".py"),
            ]
            
            for alt_path in alternatives:
                alt_full_path = backend_path / alt_path
                if alt_full_path.exists():
                    full_path = alt_full_path
                    script_path = alt_path
                    break
        
        if not full_path.exists():
            logger.error(f"❌ {name} Engine: Script not found at {full_path}")
            return False
        
        logger.info(f"🚀 Starting {name} Engine on port {port}...")
        
        # Start the engine process
        process = subprocess.Popen(
            [sys.executable, str(full_path)],
            cwd=str(backend_path),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info(f"✅ {name} Engine started successfully (PID: {process.pid})")
            return True
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ {name} Engine failed to start")
            if stderr:
                logger.error(f"Error: {stderr.decode()[:200]}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to start {name} Engine: {e}")
        return False

def main():
    """Start all engines with M4 Max optimization"""
    logger.info("🔧 Starting All Engines with M4 Max Optimization")
    logger.info("=" * 60)
    
    # Verify M4 Max detection
    from universal_m4_max_detection import is_m4_max_detected, enable_m4_max_optimization
    
    if not is_m4_max_detected():
        logger.info("🚀 Enabling M4 Max optimization...")
        enable_m4_max_optimization()
    
    if is_m4_max_detected():
        logger.info("✅ M4 Max hardware optimization ENABLED")
    else:
        logger.warning("⚠️  M4 Max not detected, proceeding anyway")
    
    logger.info("")
    
    # Start all engines
    results = {}
    for name, port, script_path in ENGINES:
        success = start_engine(name, port, script_path)
        results[name] = {"port": port, "success": success}
        time.sleep(1)  # Stagger starts
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ENGINE STARTUP SUMMARY")
    logger.info("=" * 60)
    
    successful_engines = 0
    for name, result in results.items():
        status = "✅ RUNNING" if result["success"] else "❌ FAILED"
        logger.info(f"{name:15} (Port {result['port']:5}): {status}")
        if result["success"]:
            successful_engines += 1
    
    logger.info(f"\nEngines Started: {successful_engines}/{len(ENGINES)}")
    
    if successful_engines == len(ENGINES):
        logger.info("🎉 ALL ENGINES STARTED SUCCESSFULLY WITH M4 MAX OPTIMIZATION!")
    else:
        logger.warning(f"⚠️  {len(ENGINES) - successful_engines} engines failed to start")
    
    return results

if __name__ == "__main__":
    results = main()