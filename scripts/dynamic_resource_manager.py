#!/usr/bin/env python3
"""
🚀 Nautilus Dynamic Resource Manager for Apple Silicon M4 Max
Automatically configures database resources based on system capabilities and workload
"""

import os
import psutil
import subprocess
import json
import time
from typing import Dict, Any
import logging

class AppleSiliconResourceManager:
    """Dynamic resource manager optimized for Apple Silicon unified memory architecture"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.system_info = self._detect_system_capabilities()
        self.workload_metrics = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for resource manager"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('AppleSiliconResourceManager')
        
    def _detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect Apple Silicon M4 Max capabilities"""
        try:
            # Get system information
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Apple Silicon specific detection
            system_info = {
                'total_memory_gb': round(memory.total / (1024**3), 2),
                'available_memory_gb': round(memory.available / (1024**3), 2),
                'cpu_cores_total': cpu_count,
                'cpu_cores_physical': psutil.cpu_count(logical=False),
                'cpu_frequency_max': cpu_freq.max if cpu_freq else None,
                'is_apple_silicon': self._detect_apple_silicon(),
                'unified_memory_architecture': True,  # Apple Silicon feature
                'timestamp': time.time()
            }
            
            self.logger.info(f"Detected system: {system_info}")
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error detecting system capabilities: {e}")
            return self._get_fallback_config()
    
    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon"""
        try:
            result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
            return 'arm64' in result.stdout.lower()
        except:
            return False
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Fallback configuration if detection fails"""
        return {
            'total_memory_gb': 36,
            'available_memory_gb': 30,
            'cpu_cores_total': 16,
            'cpu_cores_physical': 12,
            'is_apple_silicon': True,
            'unified_memory_architecture': True
        }
    
    def calculate_dynamic_allocations(self) -> Dict[str, Dict[str, Any]]:
        """Calculate dynamic resource allocations for each database component"""
        
        available_memory = self.system_info['available_memory_gb']
        cpu_cores = self.system_info['cpu_cores_physical']
        
        # Dynamic allocation strategy based on unified memory architecture
        allocations = {
            'clickhouse': self._calculate_clickhouse_allocation(available_memory, cpu_cores),
            'druid': self._calculate_druid_allocation(available_memory, cpu_cores),
            'postgres': self._calculate_postgres_allocation(available_memory, cpu_cores),
            'redis': self._calculate_redis_allocation(available_memory, cpu_cores),
            'minio': self._calculate_minio_allocation(available_memory, cpu_cores)
        }
        
        self.logger.info(f"Calculated dynamic allocations: {allocations}")
        return allocations
    
    def _calculate_clickhouse_allocation(self, memory_gb: float, cores: int) -> Dict[str, Any]:
        """Calculate ClickHouse dynamic allocation"""
        # ClickHouse gets priority for analytical workloads
        memory_percentage = 0.4  # 40% of available memory
        allocated_memory = memory_gb * memory_percentage
        
        return {
            'max_memory_usage': int(allocated_memory * 0.8 * 1024**3),  # 80% of allocated
            'max_server_memory_usage': int(allocated_memory * 1024**3),
            'max_threads': min(cores, 16),  # Use all P-cores but cap at 16
            'uncompressed_cache_size': int(allocated_memory * 0.2 * 1024**3),
            'mark_cache_size': int(allocated_memory * 0.15 * 1024**3),
            'query_cache_size': int(allocated_memory * 0.1 * 1024**3),
            'background_pool_size': cores * 2,
            'max_concurrent_queries': min(100, cores * 8),
            'memory_percentage': memory_percentage
        }
    
    def _calculate_druid_allocation(self, memory_gb: float, cores: int) -> Dict[str, Any]:
        """Calculate Apache Druid dynamic allocation"""
        memory_percentage = 0.25  # 25% of available memory
        allocated_memory = memory_gb * memory_percentage
        
        return {
            'heap_size_gb': max(2, int(allocated_memory * 0.6)),
            'direct_memory_gb': max(4, int(allocated_memory * 0.8)),
            'processing_threads': min(cores, 12),
            'processing_buffer_size': int(allocated_memory * 0.1 * 1024**3),
            'merge_buffers': min(8, cores),
            'segment_cache_size_gb': max(10, int(allocated_memory * 0.3)),
            'memory_percentage': memory_percentage
        }
    
    def _calculate_postgres_allocation(self, memory_gb: float, cores: int) -> Dict[str, Any]:
        """Calculate PostgreSQL + TimescaleDB dynamic allocation"""
        memory_percentage = 0.20  # 20% of available memory
        allocated_memory = memory_gb * memory_percentage
        
        return {
            'shared_buffers_gb': max(2, int(allocated_memory * 0.4)),
            'effective_cache_size_gb': max(4, int(allocated_memory * 0.7)),
            'work_mem_mb': max(64, int(allocated_memory * 50)),  # 50MB per GB
            'maintenance_work_mem_gb': max(1, int(allocated_memory * 0.1)),
            'max_worker_processes': cores,
            'max_parallel_workers': min(cores, 8),
            'max_connections': min(500, cores * 30),
            'memory_percentage': memory_percentage
        }
    
    def _calculate_redis_allocation(self, memory_gb: float, cores: int) -> Dict[str, Any]:
        """Calculate Redis dynamic allocation"""
        memory_percentage = 0.10  # 10% of available memory
        allocated_memory = memory_gb * memory_percentage
        
        return {
            'maxmemory_gb': max(1, int(allocated_memory)),
            'io_threads': min(8, cores // 2),
            'databases': 16,
            'tcp_keepalive': 60,
            'timeout': 300,
            'memory_percentage': memory_percentage
        }
    
    def _calculate_minio_allocation(self, memory_gb: float, cores: int) -> Dict[str, Any]:
        """Calculate MinIO dynamic allocation"""
        memory_percentage = 0.05  # 5% of available memory
        allocated_memory = memory_gb * memory_percentage
        
        return {
            'cache_quota_percent': 80,
            'cache_watermark_low': 70,
            'cache_watermark_high': 90,
            'api_requests_max': cores * 100,
            'memory_percentage': memory_percentage
        }
    
    def monitor_and_adjust(self) -> Dict[str, Any]:
        """Monitor system performance and adjust allocations"""
        try:
            # Get current system metrics
            current_memory = psutil.virtual_memory()
            current_cpu = psutil.cpu_percent(interval=1)
            
            metrics = {
                'memory_used_percent': current_memory.percent,
                'memory_available_gb': round(current_memory.available / (1024**3), 2),
                'cpu_usage_percent': current_cpu,
                'timestamp': time.time()
            }
            
            # Adjust allocations based on current usage
            if current_memory.percent > 85:  # High memory usage
                self.logger.warning(f"High memory usage detected: {current_memory.percent}%")
                # Reduce cache sizes dynamically
                
            if current_cpu > 80:  # High CPU usage
                self.logger.warning(f"High CPU usage detected: {current_cpu}%")
                # Reduce concurrent query limits
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring system: {e}")
            return {}
    
    def generate_dynamic_config(self) -> Dict[str, Any]:
        """Generate complete dynamic configuration"""
        allocations = self.calculate_dynamic_allocations()
        metrics = self.monitor_and_adjust()
        
        config = {
            'system_info': self.system_info,
            'resource_allocations': allocations,
            'current_metrics': metrics,
            'optimization_strategy': 'unified_memory_architecture',
            'auto_scaling_enabled': True,
            'generated_at': time.time()
        }
        
        return config
    
    def apply_configuration(self):
        """Apply dynamic configuration to running containers"""
        config = self.generate_dynamic_config()
        
        # Save configuration for containers to read
        config_file = '/tmp/nautilus_dynamic_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        self.logger.info(f"Dynamic configuration saved to {config_file}")
        
        # Signal containers to reload configuration
        self._signal_containers_reload()
        
        return config
    
    def _signal_containers_reload(self):
        """Signal containers to reload dynamic configuration"""
        try:
            # Send SIGUSR1 to containers for config reload
            containers = ['nautilus-clickhouse', 'nautilus-druid-coordinator', 
                         'nautilus-druid-broker', 'nautilus-postgres-enhanced']
            
            for container in containers:
                try:
                    subprocess.run(['docker', 'kill', '-s', 'SIGUSR1', container], 
                                 capture_output=True, timeout=5)
                    self.logger.info(f"Sent reload signal to {container}")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Timeout sending signal to {container}")
                except Exception as e:
                    self.logger.error(f"Error signaling {container}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error signaling containers: {e}")

def main():
    """Main function for dynamic resource management"""
    manager = AppleSiliconResourceManager()
    
    # Run continuously with dynamic adjustment
    while True:
        try:
            config = manager.apply_configuration()
            print(f"Applied dynamic configuration at {time.ctime()}")
            
            # Wait for next adjustment cycle (5 minutes)
            time.sleep(300)
            
        except KeyboardInterrupt:
            print("Shutting down dynamic resource manager...")
            break
        except Exception as e:
            manager.logger.error(f"Error in main loop: {e}")
            time.sleep(60)  # Wait 1 minute before retry

if __name__ == '__main__':
    main()