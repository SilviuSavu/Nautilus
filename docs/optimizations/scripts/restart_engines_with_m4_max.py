#!/usr/bin/env python3
"""
Engine Restart Script with M4 Max Acceleration
Restarts engines with proper M4 Max hardware acceleration enabled
"""

import os
import sys
import time
import signal
import subprocess
import asyncio
import logging
from typing import List, Dict, Any
import requests

logger = logging.getLogger(__name__)

class EngineRestartManager:
    """Manages engine restart with M4 Max acceleration"""
    
    def __init__(self):
        self.engines_to_restart = [
            {"name": "Factor Engine", "port": 8300, "path": "engines/factor/ultra_fast_factor_engine.py"},
            {"name": "WebSocket Engine", "port": 8600, "path": "engines/websocket/ultra_fast_websocket_engine.py"},
            {"name": "Portfolio Engine", "port": 8900, "path": "engines/portfolio/ultra_fast_portfolio_engine.py"}
        ]
        self.started_processes = []
        
    def enable_m4_max_optimization(self):
        """Enable M4 Max optimization globally"""
        os.environ['M4_MAX_OPTIMIZED'] = '1'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable PyTorch Metal
        os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'    # Enable Metal GPU
        
        print("✅ M4 Max optimization environment variables set:")
        print(f"   M4_MAX_OPTIMIZED = {os.environ.get('M4_MAX_OPTIMIZED')}")
        print(f"   PYTORCH_ENABLE_MPS_FALLBACK = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
        print(f"   METAL_DEVICE_WRAPPER_TYPE = {os.environ.get('METAL_DEVICE_WRAPPER_TYPE')}")
    
    def find_engine_pids(self, port: int) -> List[int]:
        """Find PIDs of processes listening on a port"""
        try:
            result = subprocess.run(['lsof', '-i', f':{port}'], 
                                 capture_output=True, text=True)
            pids = []
            for line in result.stdout.split('\n'):
                if 'LISTEN' in line and 'python' in line.lower():
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pids.append(int(parts[1]))
                        except ValueError:
                            continue
            return pids
        except Exception as e:
            print(f"❌ Error finding PIDs for port {port}: {e}")
            return []
    
    def stop_engine(self, engine: Dict[str, Any]) -> bool:
        """Stop an engine by killing its processes"""
        port = engine["port"]
        name = engine["name"]
        
        pids = self.find_engine_pids(port)
        if not pids:
            print(f"✅ {name} - No processes to stop")
            return True
            
        print(f"🛑 Stopping {name} (PIDs: {pids})...")
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                # Force kill if still running
                try:
                    os.kill(pid, 0)  # Check if still running
                    os.kill(pid, signal.SIGKILL)
                    print(f"   Force killed PID {pid}")
                except ProcessLookupError:
                    print(f"   PID {pid} stopped gracefully")
            except ProcessLookupError:
                print(f"   PID {pid} already stopped")
            except Exception as e:
                print(f"   ❌ Error stopping PID {pid}: {e}")
        
        time.sleep(3)  # Allow cleanup
        return True
    
    def start_engine(self, engine: Dict[str, Any]) -> bool:
        """Start an engine process"""
        name = engine["name"]
        path = engine["path"]
        port = engine["port"]
        
        print(f"🚀 Starting {name}...")
        
        try:
            # Start the engine process
            process = subprocess.Popen([
                sys.executable, path
            ], cwd='/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend',
               env=os.environ.copy())
            
            self.started_processes.append(process)
            
            # Wait for startup
            print(f"   Waiting for {name} to start...")
            for attempt in range(10):  # 30 seconds max
                time.sleep(3)
                try:
                    response = requests.get(f"http://localhost:{port}/health", timeout=2)
                    if response.status_code == 200:
                        health_data = response.json()
                        status = health_data.get('status', 'unknown')
                        m4_detected = health_data.get('hardware_status', {}).get('m4_max_detected', False)
                        
                        print(f"✅ {name} started successfully!")
                        print(f"   Status: {status}")
                        print(f"   M4 Max detected: {m4_detected}")
                        return True
                except:
                    continue
                    
            print(f"❌ {name} failed to start or respond")
            return False
            
        except Exception as e:
            print(f"❌ Error starting {name}: {e}")
            return False
    
    def restart_all_engines(self):
        """Restart all engines with M4 Max acceleration"""
        print("🔧 JAMES: Restarting engines with M4 Max acceleration...")
        print("=" * 60)
        
        # Enable M4 Max optimization
        self.enable_m4_max_optimization()
        
        # Stop all engines first
        print("\n📍 Phase 1: Stopping engines...")
        for engine in self.engines_to_restart:
            self.stop_engine(engine)
        
        print("\n⏱️  Waiting for cleanup...")
        time.sleep(5)
        
        # Start all engines
        print("\n📍 Phase 2: Starting engines with M4 Max acceleration...")
        results = {}
        for engine in self.engines_to_restart:
            results[engine["name"]] = self.start_engine(engine)
        
        # Summary
        print("\n📋 RESTART SUMMARY:")
        print("=" * 40)
        for name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{name}: {status}")
        
        successful = sum(results.values())
        total = len(results)
        print(f"\nEngines restarted: {successful}/{total}")
        
        if successful == total:
            print("🎉 All engines successfully restarted with M4 Max acceleration!")
        else:
            print("⚠️  Some engines failed to restart - check logs above")
        
        return results
    
    def cleanup(self):
        """Cleanup started processes"""
        for process in self.started_processes:
            if process.poll() is None:  # Still running
                process.terminate()


def main():
    """Main restart function"""
    restart_manager = EngineRestartManager()
    
    try:
        results = restart_manager.restart_all_engines()
        return results
    except KeyboardInterrupt:
        print("\n🛑 Restart interrupted by user")
        restart_manager.cleanup()
        return {}
    except Exception as e:
        print(f"❌ Restart failed: {e}")
        restart_manager.cleanup()
        return {}


if __name__ == "__main__":
    main()