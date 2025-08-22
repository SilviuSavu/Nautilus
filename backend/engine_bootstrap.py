#!/usr/bin/env python3
"""
NautilusTrader Engine Bootstrap
Production-ready container bootstrap for real trading engine execution

Sprint 2: Container-in-Container Pattern Implementation
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/engine_bootstrap.log')
    ]
)
logger = logging.getLogger(__name__)

class EngineBootstrap:
    """Bootstrap manager for NautilusTrader engine container"""
    
    def __init__(self):
        self.app = FastAPI(title="NautilusTrader Engine", version="1.0.0")
        self.engine_process: Optional[asyncio.subprocess.Process] = None
        self.engine_config: Optional[Dict[str, Any]] = None
        self.is_shutting_down = False
        self.health_status = "starting"
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Setup routes
        self._setup_routes()
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_shutting_down = True
        asyncio.create_task(self._graceful_shutdown())
        
    async def _graceful_shutdown(self):
        """Gracefully shutdown the engine"""
        try:
            if self.engine_process and self.engine_process.returncode is None:
                logger.info("Stopping NautilusTrader engine...")
                self.engine_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(self.engine_process.wait(), timeout=30)
                    logger.info("Engine stopped gracefully")
                except asyncio.TimeoutError:
                    logger.warning("Engine didn't stop gracefully, force killing...")
                    self.engine_process.kill()
                    await self.engine_process.wait()
                    
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            self.health_status = "stopped"
            logger.info("Bootstrap shutdown complete")
            
    def _setup_routes(self):
        """Setup FastAPI routes for engine communication"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": self.health_status,
                "timestamp": datetime.now().isoformat(),
                "engine_running": self.engine_process is not None and self.engine_process.returncode is None
            }
            
        @self.app.post("/start")
        async def start_engine(config: dict):
            if self.engine_process and self.engine_process.returncode is None:
                raise HTTPException(status_code=400, detail="Engine already running")
                
            try:
                await self._start_engine(config)
                return {"success": True, "message": "Engine started"}
            except Exception as e:
                logger.error(f"Error starting engine: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/stop")
        async def stop_engine():
            if not self.engine_process or self.engine_process.returncode is not None:
                raise HTTPException(status_code=400, detail="Engine not running")
                
            try:
                await self._stop_engine()
                return {"success": True, "message": "Engine stopped"}
            except Exception as e:
                logger.error(f"Error stopping engine: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/status")
        async def get_status():
            return {
                "health_status": self.health_status,
                "engine_running": self.engine_process is not None and self.engine_process.returncode is None,
                "config": self.engine_config,
                "timestamp": datetime.now().isoformat()
            }
            
    async def _start_engine(self, config: Dict[str, Any]):
        """Start the NautilusTrader engine with configuration"""
        try:
            logger.info("Starting NautilusTrader engine...")
            self.engine_config = config
            
            # Create configuration file
            config_path = await self._create_config_file(config)
            
            # Prepare engine command
            engine_cmd = [
                sys.executable, "-m", "nautilus_trader.live.node",
                "--config", config_path
            ]
            
            # Start engine process
            self.engine_process = await asyncio.create_subprocess_exec(
                *engine_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/app"
            )
            
            self.health_status = "running"
            logger.info(f"Engine started with PID: {self.engine_process.pid}")
            
            # Monitor engine process
            asyncio.create_task(self._monitor_engine())
            
        except Exception as e:
            self.health_status = "error"
            logger.error(f"Failed to start engine: {e}")
            raise
            
    async def _stop_engine(self):
        """Stop the NautilusTrader engine"""
        if self.engine_process:
            logger.info("Stopping NautilusTrader engine...")
            self.engine_process.terminate()
            
            try:
                await asyncio.wait_for(self.engine_process.wait(), timeout=30)
                logger.info("Engine stopped gracefully")
            except asyncio.TimeoutError:
                logger.warning("Force killing engine...")
                self.engine_process.kill()
                await self.engine_process.wait()
                
            self.engine_process = None
            self.health_status = "stopped"
            
    async def _monitor_engine(self):
        """Monitor engine process and handle output"""
        if not self.engine_process:
            return
            
        try:
            while True:
                line = await self.engine_process.stdout.readline()
                if not line:
                    break
                    
                log_line = line.decode().strip()
                if log_line:
                    logger.info(f"ENGINE: {log_line}")
                    
        except Exception as e:
            logger.error(f"Error monitoring engine: {e}")
        finally:
            if self.engine_process and self.engine_process.returncode is not None:
                self.health_status = "stopped" if self.engine_process.returncode == 0 else "error"
                logger.info(f"Engine process exited with code: {self.engine_process.returncode}")
                
    async def _create_config_file(self, config: Dict[str, Any]) -> str:
        """Create NautilusTrader configuration file"""
        config_dir = Path("/app/config")
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "engine_config.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Created configuration file: {config_file}")
        return str(config_file)
        
    async def run_standby_mode(self):
        """Run in standby mode - ready to receive commands"""
        logger.info("NautilusTrader Engine Bootstrap starting in standby mode...")
        self.health_status = "ready"
        
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=8001,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    async def run_direct_mode(self, config_file: str):
        """Run engine directly with configuration file"""
        logger.info(f"Starting engine in direct mode with config: {config_file}")
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        await self._start_engine(config)
        
        # Wait for engine to complete
        if self.engine_process:
            await self.engine_process.wait()
            
    def perform_health_check(self) -> bool:
        """Perform health check"""
        try:
            # Basic imports test
            import nautilus_trader
            
            # Check critical paths
            required_paths = [
                "/app/config",
                "/app/data", 
                "/app/cache",
                "/app/results",
                "/app/logs"
            ]
            
            for path in required_paths:
                if not os.path.exists(path):
                    logger.error(f"Required path missing: {path}")
                    return False
                    
            logger.info("Health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="NautilusTrader Engine Bootstrap")
    parser.add_argument("--mode", choices=["standby", "direct"], default="standby",
                       help="Bootstrap mode")
    parser.add_argument("--config", help="Configuration file for direct mode")
    parser.add_argument("--health-check", action="store_true", 
                       help="Perform health check and exit")
    
    args = parser.parse_args()
    
    bootstrap = EngineBootstrap()
    
    if args.health_check:
        success = bootstrap.perform_health_check()
        sys.exit(0 if success else 1)
        
    try:
        if args.mode == "direct":
            if not args.config:
                logger.error("Direct mode requires --config argument")
                sys.exit(1)
            await bootstrap.run_direct_mode(args.config)
        else:
            await bootstrap.run_standby_mode()
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Bootstrap error: {e}")
        sys.exit(1)
    finally:
        logger.info("Bootstrap shutdown")

if __name__ == "__main__":
    asyncio.run(main())