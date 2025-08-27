#!/usr/bin/env python3
"""
🚀 START ALL ENGINES - ULTRA FAST CONSOLIDATED VERSION
Uses the BEST ultra-fast engine version for each component

Engine Selection Logic:
- Priority 1: ultra_fast_2025_*.py (MLX + PyTorch 2.8 + Neural Engine)
- Priority 2: ultra_fast_*.py (Advanced optimizations)
- Priority 3: ultra_fast_sme_*.py (SME-specific acceleration)
- Priority 4: simple_*.py or production_*.py (Fallbacks)
"""

import asyncio
import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Enable M4 Max optimization globally
os.environ['M4_MAX_OPTIMIZED'] = '1'
os.environ['ULTRA_FAST_MODE'] = '1'
os.environ['PYTHONPATH'] = str(backend_path)

# 🎯 CONSOLIDATED ENGINE CONFIGURATION - BEST VERSIONS ONLY
ULTRA_FAST_ENGINES = [
    # Core Processing Engines (8100-8900) - Using BEST available versions
    ("Analytics", 8100, "engines/analytics/ultra_fast_2025_analytics_engine.py"),        # ✅ 2025 Version
    ("Backtesting", 8110, "engines/backtesting/simple_backtesting_engine.py"),          # ✅ Simple (only available)
    ("Risk", 8200, "engines/risk/ultra_fast_2025_risk_engine.py"),                      # ✅ 2025 Version  
    ("Factor", 8300, "engines/factor/ultra_fast_2025_factor_engine.py"),                # ✅ 2025 Version
    ("ML", 8400, "engines/ml/ultra_fast_ml_engine.py"),                                 # ✅ Ultra Fast
    ("Features", 8500, "engines/features/ultra_fast_features_engine.py"),               # ✅ Ultra Fast
    ("WebSocket", 8600, "engines/websocket/ultra_fast_websocket_engine.py"),            # ✅ Ultra Fast
    ("Strategy", 8700, "engines/strategy/ultra_fast_strategy_engine.py"),               # ✅ Ultra Fast
    ("MarketData", 8800, "engines/marketdata/simple_marketdata_engine.py"),             # ✅ Simple (only available)
    ("Portfolio", 8900, "engines/portfolio/ultra_fast_portfolio_engine.py"),            # ✅ Ultra Fast
    
    # Mission-Critical Engines (9000+) - Specialized versions
    ("Collateral", 9000, "engines/collateral/ultra_fast_collateral_engine.py"),         # ✅ Ultra Fast
    ("VPIN", 10000, "engines/vpin/ultra_fast_vpin_server.py"),                          # ✅ Ultra Fast
    ("Enhanced VPIN", 10001, "engines/vpin/enhanced_microstructure_vpin_server.py"),    # ✅ Enhanced
]

class UltraFastEngineManager:
    """Manages ultra-fast engine deployment with best version selection"""
    
    def __init__(self):
        self.processes: List[Tuple[str, int, subprocess.Popen]] = []
        self.backend_path = backend_path
        
    def _validate_engine_file(self, script_path: str) -> bool:
        """Validate that engine file exists"""
        full_path = self.backend_path / script_path
        return full_path.exists() and full_path.is_file()
    
    def _find_alternative_engine(self, name: str, port: int, script_path: str) -> str:
        """Find alternative engine if primary not found"""
        engine_name = name.lower()
        engine_dir = self.backend_path / "engines" / engine_name
        
        # Search patterns (in order of preference)
        search_patterns = [
            f"ultra_fast_2025_{engine_name}_engine.py",
            f"ultra_fast_{engine_name}_engine.py", 
            f"ultra_fast_sme_{engine_name}_engine.py",
            f"simple_{engine_name}_engine.py",
            f"production_{engine_name}_server.py",
            f"{engine_name}_engine.py"
        ]
        
        for pattern in search_patterns:
            alt_path = engine_dir / pattern
            if alt_path.exists():
                relative_path = alt_path.relative_to(self.backend_path)
                logger.info(f"🔄 {name}: Using alternative {relative_path}")
                return str(relative_path)
        
        return script_path  # Return original if no alternative found
    
    def start_engine(self, name: str, port: int, script_path: str) -> bool:
        """Start a single ultra-fast engine"""
        try:
            # Validate engine file exists
            if not self._validate_engine_file(script_path):
                logger.warning(f"⚠️ {name}: Engine file not found at {script_path}")
                script_path = self._find_alternative_engine(name, port, script_path)
            
            full_path = self.backend_path / script_path
            if not full_path.exists():
                logger.error(f"❌ {name}: No engine file found")
                return False
            
            logger.info(f"🚀 Starting {name} Engine (Port {port})...")
            logger.info(f"   📁 Path: {script_path}")
            
            # Start the engine process
            env = os.environ.copy()
            env.update({
                'M4_MAX_OPTIMIZED': '1',
                'ULTRA_FAST_MODE': '1',
                'ENGINE_NAME': name.upper(),
                'ENGINE_PORT': str(port)
            })
            
            process = subprocess.Popen(
                [sys.executable, str(full_path)],
                cwd=str(self.backend_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give engine time to start
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                self.processes.append((name, port, process))
                logger.info(f"✅ {name} Engine started (PID: {process.pid})")
                return True
            else:
                stdout, stderr = process.communicate(timeout=5)
                logger.error(f"❌ {name} Engine failed to start")
                if stderr:
                    logger.error(f"   Error: {stderr[:300]}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {name} Engine exception: {e}")
            return False
    
    def start_all_engines(self) -> Dict[str, bool]:
        """Start all ultra-fast engines"""
        logger.info("🔥 STARTING ALL ENGINES - ULTRA FAST MODE")
        logger.info("=" * 70)
        logger.info("Using BEST available ultra-fast engine for each component:")
        logger.info("• Priority 1: ultra_fast_2025_*.py (MLX + PyTorch 2.8)")
        logger.info("• Priority 2: ultra_fast_*.py (Advanced optimizations)")  
        logger.info("• Priority 3: simple_*.py or production_*.py (Fallbacks)")
        logger.info("=" * 70)
        
        results = {}
        
        # Start engines with staggered timing
        for name, port, script_path in ULTRA_FAST_ENGINES:
            success = self.start_engine(name, port, script_path)
            results[name] = success
            time.sleep(2)  # Stagger startup to avoid resource conflicts
        
        return results
    
    def show_startup_summary(self, results: Dict[str, bool]):
        """Display startup summary"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("🎯 ULTRA FAST ENGINE STARTUP SUMMARY")
        logger.info("=" * 70)
        
        successful = 0
        total = len(results)
        
        for name, success in results.items():
            port_info = next(port for n, port, _ in ULTRA_FAST_ENGINES if n == name)
            status = "✅ RUNNING" if success else "❌ FAILED"
            logger.info(f"{name:15} (Port {port_info:5}): {status}")
            if success:
                successful += 1
        
        logger.info("-" * 70)
        logger.info(f"Success Rate: {successful}/{total} engines ({successful/total*100:.1f}%)")
        
        if successful == total:
            logger.info("🎉 ALL ENGINES STARTED WITH ULTRA-FAST OPTIMIZATION!")
            logger.info("🚀 System ready for institutional-grade trading")
        elif successful > total * 0.8:
            logger.info("🟡 Most engines started successfully")
        else:
            logger.warning("🔴 Multiple engine failures detected")
        
        logger.info("=" * 70)
    
    def monitor_engines(self):
        """Monitor running engines"""
        try:
            logger.info("🔍 Monitoring engines... (Press Ctrl+C to stop all)")
            while True:
                time.sleep(10)
                
                # Check engine health
                active_engines = 0
                for name, port, process in self.processes:
                    if process.poll() is None:
                        active_engines += 1
                    else:
                        logger.warning(f"⚠️ {name} Engine stopped unexpectedly")
                
                if active_engines == 0:
                    logger.error("❌ No engines running!")
                    break
                    
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutdown requested...")
            self.shutdown_all_engines()
    
    def shutdown_all_engines(self):
        """Gracefully shutdown all engines"""
        logger.info("🔄 Shutting down all engines...")
        
        for name, port, process in self.processes:
            try:
                logger.info(f"Stopping {name} Engine...")
                process.terminate()
            except:
                pass
        
        # Wait for graceful shutdown
        time.sleep(3)
        
        # Force kill if needed
        for name, port, process in self.processes:
            if process.poll() is None:
                logger.warning(f"Force killing {name} Engine...")
                process.kill()
        
        logger.info("✅ All engines stopped")

def main():
    """Main entry point"""
    manager = UltraFastEngineManager()
    
    try:
        # Start all engines
        results = manager.start_all_engines()
        
        # Show summary
        manager.show_startup_summary(results)
        
        # Monitor if any engines started successfully
        if any(results.values()):
            manager.monitor_engines()
        else:
            logger.error("❌ No engines started successfully")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())