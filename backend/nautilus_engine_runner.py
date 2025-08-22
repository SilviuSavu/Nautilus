#!/usr/bin/env python3
"""
NautilusTrader Engine Runner
Production engine execution with proper lifecycle management

Sprint 2: Real Engine Implementation
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from nautilus_trader.cache.redis import RedisCache
from nautilus_trader.common.component import Logger
from nautilus_trader.common.enums import LogLevel
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode

logger = logging.getLogger(__name__)

class NautilusEngineRunner:
    """Production NautilusTrader engine runner"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.node: Optional[TradingNode] = None
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
        
    async def start(self):
        """Start the NautilusTrader engine"""
        try:
            logger.info("Loading engine configuration...")
            config = await self._load_config()
            
            logger.info("Creating trading node...")
            self.node = TradingNode(config=config)
            
            logger.info("Starting trading node...")
            await self.node.start_async()
            self.is_running = True
            
            logger.info("NautilusTrader engine started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Error starting engine: {e}")
            raise
        finally:
            await self._cleanup()
            
    async def _load_config(self) -> TradingNodeConfig:
        """Load and validate engine configuration"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config_dict = json.load(f)
            
        # Validate and create TradingNodeConfig
        try:
            config = TradingNodeConfig.parse_obj(config_dict)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Invalid configuration: {e}")
            raise
            
    async def _cleanup(self):
        """Cleanup resources"""
        if self.node and self.is_running:
            try:
                logger.info("Stopping trading node...")
                await self.node.stop_async()
                logger.info("Trading node stopped")
            except Exception as e:
                logger.error(f"Error stopping node: {e}")
        
        self.is_running = False
        logger.info("Engine cleanup complete")

async def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: nautilus_engine_runner.py <config_file>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/app/logs/engine.log')
        ]
    )
    
    runner = NautilusEngineRunner(config_path)
    
    try:
        await runner.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt")
    except Exception as e:
        logger.error(f"Engine error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())