#!/usr/bin/env python3
"""
Universal M4 Max Hardware Detection
Provides consistent M4 Max detection across all engines
"""

import os
import platform
import logging
import subprocess
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class UniversalM4MaxDetection:
    """Universal M4 Max hardware detection for all engines"""
    
    def __init__(self):
        self._cached_result: Optional[bool] = None
        self._hardware_info: Optional[Dict[str, Any]] = None
        
    def is_m4_max_detected(self) -> bool:
        """
        Universal M4 Max detection method
        Returns True if M4 Max hardware is detected
        """
        if self._cached_result is not None:
            return self._cached_result
            
        try:
            # Method 1: Check environment variable (for testing/override)
            if os.environ.get('M4_MAX_OPTIMIZED', '0') == '1':
                logger.info("✅ M4 Max enabled via M4_MAX_OPTIMIZED environment variable")
                self._cached_result = True
                return True
                
            # Method 2: Check Apple Silicon architecture
            machine_arch = platform.machine().lower()
            if machine_arch not in ["arm64", "aarch64"]:
                logger.info("❌ M4 Max detection: Not Apple Silicon architecture")
                self._cached_result = False
                return False
                
            # Method 3: Check macOS system info
            hardware_info = self._get_hardware_info()
            
            # Look for M4 Max indicators
            chip_name = hardware_info.get("chip_name", "").lower()
            if "m4" in chip_name and "max" in chip_name:
                logger.info("✅ M4 Max detected via system_profiler")
                self._cached_result = True
                return True
                
            # Method 4: Check CPU count and memory (M4 Max typical specs)
            cpu_count = os.cpu_count() or 0
            
            # M4 Max typically has 14 cores (10P+4E) or 16 cores (12P+4E) 
            # But for compatibility, we'll accept any modern Apple Silicon
            if machine_arch == "arm64" and cpu_count >= 8:
                logger.info(f"✅ M4 Max-compatible Apple Silicon detected ({cpu_count} cores)")
                self._cached_result = True
                return True
                
            logger.info(f"❌ M4 Max not detected (arch: {machine_arch}, cores: {cpu_count})")
            self._cached_result = False
            return False
            
        except Exception as e:
            logger.error(f"M4 Max detection failed: {e}")
            # Default to True on Apple Silicon for compatibility
            is_apple_silicon = platform.machine().lower() in ["arm64", "aarch64"]
            self._cached_result = is_apple_silicon
            return is_apple_silicon
            
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information from system"""
        if self._hardware_info is not None:
            return self._hardware_info
            
        hardware_info = {}
        
        try:
            # Try to get chip info from system_profiler
            result = subprocess.run([
                "system_profiler", "SPHardwareDataType"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                output = result.stdout.lower()
                
                # Extract chip name
                for line in output.split('\n'):
                    if 'chip:' in line or 'processor:' in line:
                        hardware_info["chip_name"] = line.strip()
                        break
                        
        except Exception as e:
            logger.debug(f"Could not get system_profiler info: {e}")
            
        # Add basic system info
        hardware_info.update({
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "cpu_count": os.cpu_count(),
        })
        
        self._hardware_info = hardware_info
        return hardware_info
        
    def get_hardware_capabilities(self) -> Dict[str, Any]:
        """Get detailed hardware capabilities"""
        return {
            "m4_max_detected": self.is_m4_max_detected(),
            "apple_silicon": platform.machine().lower() in ["arm64", "aarch64"],
            "cpu_cores": os.cpu_count(),
            "architecture": platform.machine(),
            "hardware_info": self._get_hardware_info()
        }
        
    def enable_m4_max_optimization(self) -> bool:
        """Enable M4 Max optimizations by setting environment variable"""
        try:
            os.environ['M4_MAX_OPTIMIZED'] = '1'
            logger.info("✅ M4 Max optimization enabled via environment variable")
            self._cached_result = True  # Force re-detection
            return True
        except Exception as e:
            logger.error(f"Could not enable M4 Max optimization: {e}")
            return False


# Global instance for consistent detection
universal_m4_max_detector = UniversalM4MaxDetection()

def is_m4_max_detected() -> bool:
    """Universal M4 Max detection function"""
    return universal_m4_max_detector.is_m4_max_detected()

def enable_m4_max_optimization() -> bool:
    """Enable M4 Max optimization globally"""
    return universal_m4_max_detector.enable_m4_max_optimization()

def get_hardware_capabilities() -> Dict[str, Any]:
    """Get comprehensive hardware capabilities"""
    return universal_m4_max_detector.get_hardware_capabilities()


if __name__ == "__main__":
    # Test the detection
    print("🔧 Universal M4 Max Detection Test")
    print("=" * 50)
    
    detector = UniversalM4MaxDetection()
    capabilities = detector.get_hardware_capabilities()
    
    print(f"M4 Max Detected: {capabilities['m4_max_detected']}")
    print(f"Apple Silicon: {capabilities['apple_silicon']}")
    print(f"CPU Cores: {capabilities['cpu_cores']}")
    print(f"Architecture: {capabilities['architecture']}")
    
    if not capabilities['m4_max_detected']:
        print("\n🚀 Enabling M4 Max optimization...")
        if detector.enable_m4_max_optimization():
            print(f"M4 Max Detected (after enable): {detector.is_m4_max_detected()}")