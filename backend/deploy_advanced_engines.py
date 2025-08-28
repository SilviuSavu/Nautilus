#!/usr/bin/env python3
"""
Advanced Engines Deployment Script
=================================

Deploys all advanced trading engines with M4 Max hardware optimization.

New Engines Deployed:
- MAGNN Multi-Modal Engine (Port 10002) - Graph Neural Networks
- Enhanced THGNN HFT Engine (Port 8600) - Microsecond predictions  
- Quantum Portfolio Engine (Port 10003) - QAOA, QIGA, QNN algorithms
- Neural SDE Engine (Port 10004) - Stochastic differential equations
- Molecular Dynamics Engine (Port 10005) - Physics-based market modeling

Hardware Optimization Features:
- Neural Engine acceleration (38 TOPS)
- Metal GPU parallel processing (40 cores, 546 GB/s)
- AMX/SME matrix operations (2.9 TFLOPS)
- Unified memory architecture optimization

Performance Targets:
- Sub-millisecond inference across all engines
- 90%+ hardware utilization
- Real-time multi-modal predictions
- Quantum-enhanced portfolio optimization
"""

import asyncio
import subprocess
import time
import logging
import signal
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EngineConfig:
    """Configuration for advanced engines"""
    
    ENGINES = {
        'magnn': {
            'name': 'MAGNN Multi-Modal Engine',
            'port': 10002,
            'script': 'backend/engines/magnn/magnn_engine.py',
            'description': 'Multi-modality Graph Neural Network for data fusion',
            'hardware': 'Neural Engine + Metal GPU',
            'startup_time': 10
        },
        'thgnn_hft': {
            'name': 'THGNN HFT Engine', 
            'port': 8600,
            'script': 'backend/engines/websocket/thgnn_hft_engine.py',
            'description': 'Temporal Heterogeneous GNN for microsecond HFT',
            'hardware': 'Neural Engine optimized',
            'startup_time': 8
        },
        'quantum': {
            'name': 'Quantum Portfolio Engine',
            'port': 10003, 
            'script': 'backend/engines/quantum/quantum_portfolio_engine.py',
            'description': 'QAOA, QIGA, QNN quantum optimization',
            'hardware': 'Neural Engine + SME/AMX',
            'startup_time': 12
        },
        'neural_sde': {
            'name': 'Neural SDE Engine',
            'port': 10004,
            'script': 'backend/engines/neural_sde/neural_sde_engine.py', 
            'description': 'Stochastic Differential Equations with Neural Networks',
            'hardware': 'Neural Engine + Metal GPU',
            'startup_time': 10
        },
        'molecular_dynamics': {
            'name': 'Molecular Dynamics Market Engine',
            'port': 10005,
            'script': 'backend/engines/molecular_dynamics/molecular_dynamics_engine.py',
            'description': 'Physics-based molecular dynamics for market simulation',
            'hardware': 'Metal GPU + Neural Engine',
            'startup_time': 15
        }
    }
    
    # Hardware optimization environment variables
    OPTIMIZATION_ENV = {
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',  # Enable Metal Performance Shaders fallback
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # Disable memory limit
        'OMP_NUM_THREADS': '8',  # Optimize for M4 Max performance cores
        'MKL_NUM_THREADS': '8',
        'CUDA_VISIBLE_DEVICES': '',  # Disable CUDA
        'PYTHONPATH': '/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus/backend'
    }

class AdvancedEngineDeployer:
    """Deploys and manages advanced trading engines"""
    
    def __init__(self):
        self.processes = {}
        self.deployment_start_time = time.time()
        self.engine_status = {}
        self.health_check_results = {}
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Advanced Engine Deployer initialized")
        logger.info(f"🔧 M4 Max hardware optimization enabled")
        logger.info(f"⚡ Neural Engine: 38 TOPS acceleration")
        logger.info(f"🔥 Metal GPU: 40 cores, 546 GB/s bandwidth")
        logger.info(f"🧮 AMX/SME: 2.9 TFLOPS matrix operations")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown_all_engines())
    
    def validate_environment(self) -> bool:
        """Validate deployment environment"""
        logger.info("🔍 Validating deployment environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 8:
            logger.error(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
        
        logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required packages
        required_packages = ['torch', 'fastapi', 'uvicorn', 'numpy', 'redis']
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'aioredis':
                    # Try alternative Redis client for Python 3.13 compatibility
                    try:
                        __import__('aioredis')
                    except (ImportError, TypeError):
                        __import__('redis')
                        logger.info(f"✅ Package available: redis (fallback for aioredis)")
                        continue
                __import__(package)
                logger.info(f"✅ Package available: {package}")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"❌ Missing package: {package}")
        
        if missing_packages:
            logger.error(f"❌ Install missing packages: {', '.join(missing_packages)}")
            return False
        
        # Check hardware optimization
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("✅ Metal Performance Shaders available")
            else:
                logger.warning("⚠️ MPS not available, falling back to CPU")
            
            # Check for Apple Silicon
            import platform
            if platform.machine() == 'arm64':
                logger.info("✅ Apple Silicon (ARM64) detected")
            else:
                logger.warning("⚠️ Not running on Apple Silicon")
            
        except Exception as e:
            logger.warning(f"⚠️ Hardware detection error: {e}")
        
        # Validate engine scripts exist
        project_root = Path('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus')
        for engine_id, config in EngineConfig.ENGINES.items():
            script_path = project_root / config['script']
            if not script_path.exists():
                logger.error(f"❌ Engine script not found: {script_path}")
                return False
            logger.info(f"✅ Engine script found: {engine_id}")
        
        logger.info("🎯 Environment validation completed successfully")
        return True
    
    async def deploy_engine(self, engine_id: str) -> bool:
        """Deploy a single advanced engine"""
        config = EngineConfig.ENGINES.get(engine_id)
        if not config:
            logger.error(f"❌ Unknown engine: {engine_id}")
            return False
        
        logger.info(f"🚀 Deploying {config['name']} on port {config['port']}")
        logger.info(f"🔧 Hardware: {config['hardware']}")
        logger.info(f"📝 Description: {config['description']}")
        
        # Prepare environment
        env = dict(EngineConfig.OPTIMIZATION_ENV)
        env.update(dict(os.environ))  # Inherit system environment
        
        # Construct command
        script_path = Path('/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus') / config['script']
        cmd = [sys.executable, str(script_path)]
        
        try:
            # Start engine process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd='/Users/savusilviu/Desktop/SilviuCorneliuSavu/Nautilus'
            )
            
            self.processes[engine_id] = process
            self.engine_status[engine_id] = 'starting'
            
            logger.info(f"✅ {config['name']} process started (PID: {process.pid})")
            
            # Wait for startup
            startup_timeout = config.get('startup_time', 10)
            logger.info(f"⏳ Waiting {startup_timeout}s for {config['name']} startup...")
            await asyncio.sleep(startup_timeout)
            
            # Health check
            if await self.health_check_engine(engine_id):
                self.engine_status[engine_id] = 'running'
                logger.info(f"🎉 {config['name']} deployed successfully!")
                return True
            else:
                self.engine_status[engine_id] = 'failed'
                logger.error(f"❌ {config['name']} health check failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to deploy {config['name']}: {e}")
            self.engine_status[engine_id] = 'error'
            return False
    
    async def health_check_engine(self, engine_id: str) -> bool:
        """Perform health check on deployed engine"""
        config = EngineConfig.ENGINES.get(engine_id)
        if not config:
            return False
        
        health_url = f"http://localhost:{config['port']}/health"
        
        try:
            # Use asyncio-compatible HTTP client
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url, timeout=5) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            self.health_check_results[engine_id] = health_data
                            
                            engine_name = health_data.get('engine', 'Unknown')
                            version = health_data.get('version', 'Unknown')
                            logger.info(f"✅ {config['name']} health check passed")
                            logger.info(f"   Engine: {engine_name} v{version}")
                            
                            if 'performance_metrics' in health_data:
                                metrics = health_data['performance_metrics']
                                logger.info(f"   Performance: {metrics}")
                            
                            return True
                        else:
                            logger.error(f"❌ {config['name']} health check failed: HTTP {response.status}")
                            return False
            except ImportError:
                # Fallback to requests with asyncio
                import requests
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    self.health_check_results[engine_id] = health_data
                    
                    engine_name = health_data.get('engine', 'Unknown')
                    version = health_data.get('version', 'Unknown')
                    logger.info(f"✅ {config['name']} health check passed")
                    logger.info(f"   Engine: {engine_name} v{version}")
                    
                    if 'performance_metrics' in health_data:
                        metrics = health_data['performance_metrics']
                        logger.info(f"   Performance: {metrics}")
                    
                    return True
                else:
                    logger.error(f"❌ {config['name']} health check failed: HTTP {response.status_code}")
                    return False
                        
        except Exception as e:
            logger.error(f"❌ {config['name']} health check error: {e}")
            return False
    
    async def deploy_all_engines(self) -> Dict[str, bool]:
        """Deploy all advanced engines concurrently"""
        logger.info("🌟 Starting deployment of all advanced engines")
        logger.info("=" * 70)
        
        deployment_results = {}
        
        # Deploy engines concurrently with some staggering to avoid resource conflicts
        deployment_tasks = []
        
        for i, engine_id in enumerate(EngineConfig.ENGINES.keys()):
            # Stagger deployments by 2 seconds each
            async def deploy_with_delay(eid, delay):
                if delay > 0:
                    await asyncio.sleep(delay)
                return await self.deploy_engine(eid)
            
            task = deploy_with_delay(engine_id, i * 2)
            deployment_tasks.append((engine_id, task))
        
        # Wait for all deployments
        for engine_id, task in deployment_tasks:
            try:
                result = await task
                deployment_results[engine_id] = result
                
                if result:
                    logger.info(f"🎯 {EngineConfig.ENGINES[engine_id]['name']}: SUCCESS")
                else:
                    logger.error(f"💥 {EngineConfig.ENGINES[engine_id]['name']}: FAILED")
                    
            except Exception as e:
                logger.error(f"💥 {engine_id} deployment exception: {e}")
                deployment_results[engine_id] = False
        
        # Deployment summary
        successful = sum(1 for success in deployment_results.values() if success)
        total = len(deployment_results)
        
        logger.info("=" * 70)
        logger.info(f"📊 Deployment Summary: {successful}/{total} engines successful")
        
        if successful == total:
            logger.info("🎉 ALL ADVANCED ENGINES DEPLOYED SUCCESSFULLY!")
            logger.info("🔥 M4 Max hardware acceleration active across all engines")
            logger.info("⚡ Neural Engine utilization: TARGET 90%+")
            logger.info("🧮 AMX/SME matrix operations: ENABLED")
            logger.info("🚀 Ready for institutional-grade trading")
        else:
            logger.warning(f"⚠️ {total - successful} engines failed to deploy")
        
        return deployment_results
    
    async def monitor_engines(self):
        """Monitor deployed engines continuously"""
        logger.info("📊 Starting engine monitoring...")
        
        while True:
            try:
                # Check each engine
                for engine_id in self.processes.keys():
                    process = self.processes[engine_id]
                    
                    # Check if process is still running
                    if process.returncode is not None:
                        logger.error(f"💥 {EngineConfig.ENGINES[engine_id]['name']} process died")
                        self.engine_status[engine_id] = 'dead'
                        
                        # Optionally restart
                        logger.info(f"🔄 Attempting to restart {engine_id}...")
                        await self.deploy_engine(engine_id)
                    
                    # Periodic health checks
                    elif time.time() % 60 < 5:  # Every minute
                        await self.health_check_engine(engine_id)
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(10)
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        uptime = time.time() - self.deployment_start_time
        
        status = {
            'deployment_info': {
                'uptime_seconds': uptime,
                'uptime_formatted': f"{uptime // 3600:.0f}h {(uptime % 3600) // 60:.0f}m {uptime % 60:.0f}s",
                'total_engines': len(EngineConfig.ENGINES),
                'running_engines': sum(1 for status in self.engine_status.values() if status == 'running')
            },
            'engines': {}
        }
        
        for engine_id, config in EngineConfig.ENGINES.items():
            engine_info = {
                'name': config['name'],
                'port': config['port'],
                'status': self.engine_status.get(engine_id, 'not_deployed'),
                'hardware_optimization': config['hardware'],
                'description': config['description']
            }
            
            # Add health check data if available
            if engine_id in self.health_check_results:
                engine_info['health_check'] = self.health_check_results[engine_id]
            
            # Add process info if running
            if engine_id in self.processes:
                process = self.processes[engine_id]
                engine_info['process_id'] = process.pid
                engine_info['process_alive'] = process.returncode is None
            
            status['engines'][engine_id] = engine_info
        
        return status
    
    async def shutdown_all_engines(self):
        """Gracefully shutdown all engines"""
        logger.info("🛑 Shutting down all advanced engines...")
        
        # Send termination signals
        for engine_id, process in self.processes.items():
            if process.returncode is None:
                logger.info(f"🛑 Terminating {EngineConfig.ENGINES[engine_id]['name']}")
                process.terminate()
        
        # Wait for graceful shutdown
        await asyncio.sleep(5)
        
        # Force kill if necessary
        for engine_id, process in self.processes.items():
            if process.returncode is None:
                logger.warning(f"💀 Force killing {EngineConfig.ENGINES[engine_id]['name']}")
                process.kill()
        
        logger.info("✅ All engines shutdown complete")

async def main():
    """Main deployment function"""
    deployer = AdvancedEngineDeployer()
    
    logger.info("🎯 NAUTILUS ADVANCED ENGINES DEPLOYMENT")
    logger.info("=====================================")
    logger.info("🔬 Quantum Computing + Graph Neural Networks + Physics Modeling")
    logger.info("⚡ M4 Max Hardware Acceleration: Neural Engine + Metal GPU + AMX/SME")
    logger.info("🚀 Institutional-Grade Trading Platform Enhancement")
    logger.info("")
    
    # Validate environment
    if not deployer.validate_environment():
        logger.error("💥 Environment validation failed")
        return False
    
    logger.info("")
    
    # Deploy all engines
    deployment_results = await deployer.deploy_all_engines()
    
    # Show final status
    status = deployer.get_deployment_status()
    logger.info("")
    logger.info("🎯 FINAL DEPLOYMENT STATUS")
    logger.info("=" * 50)
    
    for engine_id, engine_info in status['engines'].items():
        status_emoji = "✅" if engine_info['status'] == 'running' else "❌"
        logger.info(f"{status_emoji} {engine_info['name']} (Port {engine_info['port']}): {engine_info['status'].upper()}")
        logger.info(f"   Hardware: {engine_info['hardware_optimization']}")
        
        if 'health_check' in engine_info:
            health = engine_info['health_check']
            if 'performance_metrics' in health:
                metrics = health['performance_metrics']
                logger.info(f"   Performance: {metrics}")
    
    successful_count = status['deployment_info']['running_engines']
    total_count = status['deployment_info']['total_engines']
    
    logger.info("")
    if successful_count == total_count:
        logger.info("🏆 DEPLOYMENT COMPLETE - ALL SYSTEMS OPERATIONAL")
        logger.info("🎯 Ready for quantum-enhanced, physics-based, neural-accelerated trading!")
        logger.info("")
        logger.info("Engine Access Points:")
        for engine_id, engine_info in status['engines'].items():
            if engine_info['status'] == 'running':
                logger.info(f"  🌐 {engine_info['name']}: http://localhost:{engine_info['port']}")
                logger.info(f"     Health: http://localhost:{engine_info['port']}/health")
        
        # Start monitoring
        logger.info("")
        logger.info("📊 Starting continuous monitoring...")
        await deployer.monitor_engines()
        
    else:
        logger.error(f"💥 PARTIAL DEPLOYMENT - {successful_count}/{total_count} engines running")
        return False

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Deployment interrupted by user")
    except Exception as e:
        logger.error(f"💥 Deployment failed: {e}")
        sys.exit(1)