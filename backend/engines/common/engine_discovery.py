#!/usr/bin/env python3
"""
Inter-Engine Discovery Protocol
Enables engines to discover each other, announce capabilities, and establish partnerships.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis.asyncio as redis
from enum import Enum

from .engine_identity import EngineIdentity, EngineRole, ProcessingCapability
from .nautilus_environment import get_nautilus_environment, MessageBusType


logger = logging.getLogger(__name__)


class DiscoveryEventType(Enum):
    """Types of discovery events"""
    ENGINE_ANNOUNCEMENT = "engine_announcement"
    ENGINE_HEARTBEAT = "engine_heartbeat"
    ENGINE_SHUTDOWN = "engine_shutdown"
    PARTNERSHIP_REQUEST = "partnership_request"
    PARTNERSHIP_RESPONSE = "partnership_response"
    PARTNERSHIP_ESTABLISHED = "partnership_established"
    PARTNERSHIP_TERMINATED = "partnership_terminated"
    CAPABILITY_UPDATE = "capability_update"


@dataclass
class DiscoveryMessage:
    """Message for engine discovery protocol"""
    event_type: DiscoveryEventType
    source_engine_id: str
    target_engine_id: Optional[str] = None  # None for broadcasts
    timestamp: str = ""
    payload: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.payload is None:
            self.payload = {}


@dataclass
class PartnershipProposal:
    """Proposal for establishing engine partnership"""
    proposer_engine_id: str
    target_engine_id: str
    relationship_type: str  # "primary", "secondary", "optional"
    data_flow_direction: str  # "input", "output", "bidirectional"
    proposed_message_types: List[str]
    expected_latency_ms: float
    expected_throughput: int
    reliability_requirement: float
    benefits_description: str
    proposal_id: str = ""
    
    def __post_init__(self):
        if not self.proposal_id:
            import uuid
            self.proposal_id = str(uuid.uuid4())


class EngineRegistry:
    """Registry of all discovered engines"""
    
    def __init__(self):
        self.engines: Dict[str, Dict[str, Any]] = {}
        self.partnerships: Dict[str, List[Dict[str, Any]]] = {}
        self.last_heartbeat: Dict[str, datetime] = {}
        self.offline_engines: Set[str] = set()
    
    def register_engine(self, engine_data: Dict[str, Any]):
        """Register a new engine or update existing"""
        engine_id = engine_data["engine_id"]
        self.engines[engine_id] = engine_data
        self.last_heartbeat[engine_id] = datetime.now()
        
        if engine_id in self.offline_engines:
            self.offline_engines.remove(engine_id)
            logger.info(f"Engine {engine_id} is back online")
    
    def update_heartbeat(self, engine_id: str):
        """Update heartbeat for engine"""
        if engine_id in self.engines:
            self.last_heartbeat[engine_id] = datetime.now()
            if engine_id in self.offline_engines:
                self.offline_engines.remove(engine_id)
    
    def mark_engine_offline(self, engine_id: str):
        """Mark engine as offline"""
        self.offline_engines.add(engine_id)
        logger.warning(f"Engine {engine_id} marked as offline")
    
    def get_engine(self, engine_id: str) -> Optional[Dict[str, Any]]:
        """Get engine data"""
        return self.engines.get(engine_id)
    
    def get_all_engines(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered engines"""
        return self.engines.copy()
    
    def get_online_engines(self) -> Dict[str, Dict[str, Any]]:
        """Get only online engines"""
        return {eid: data for eid, data in self.engines.items() 
                if eid not in self.offline_engines}
    
    def find_engines_by_capability(self, capability: str) -> List[str]:
        """Find engines that have specific capability"""
        result = []
        for engine_id, data in self.get_online_engines().items():
            if capability in data.get("capabilities", []):
                result.append(engine_id)
        return result
    
    def find_engines_by_role(self, role: str) -> List[str]:
        """Find engines that fulfill specific role"""
        result = []
        for engine_id, data in self.get_online_engines().items():
            if role in data.get("roles", []):
                result.append(engine_id)
        return result
    
    def get_partnership_candidates(self, engine_id: str) -> List[str]:
        """Get potential partnership candidates for engine"""
        if engine_id not in self.engines:
            return []
        
        engine_data = self.engines[engine_id]
        preferred_partners = engine_data.get("preferred_partners", [])
        
        # Filter to online engines only
        candidates = []
        for partner_id in preferred_partners:
            if partner_id in self.engines and partner_id not in self.offline_engines:
                candidates.append(partner_id)
        
        return candidates
    
    def add_partnership(self, engine1_id: str, engine2_id: str, partnership_data: Dict[str, Any]):
        """Record a partnership between two engines"""
        if engine1_id not in self.partnerships:
            self.partnerships[engine1_id] = []
        if engine2_id not in self.partnerships:
            self.partnerships[engine2_id] = []
        
        partnership_data["established_at"] = datetime.now().isoformat()
        
        self.partnerships[engine1_id].append({
            "partner_id": engine2_id,
            **partnership_data
        })
        self.partnerships[engine2_id].append({
            "partner_id": engine1_id,
            **partnership_data
        })
    
    def check_stale_engines(self, timeout_seconds: int = 60) -> List[str]:
        """Check for engines that haven't sent heartbeat recently"""
        cutoff_time = datetime.now() - timedelta(seconds=timeout_seconds)
        stale_engines = []
        
        for engine_id, last_heartbeat in self.last_heartbeat.items():
            if last_heartbeat < cutoff_time and engine_id not in self.offline_engines:
                stale_engines.append(engine_id)
                self.mark_engine_offline(engine_id)
        
        return stale_engines


class EngineDiscoveryProtocol:
    """Protocol for engine discovery and partnership establishment"""
    
    def __init__(self, engine_identity: EngineIdentity):
        self.identity = engine_identity
        self.registry = EngineRegistry()
        
        # Redis connections for different buses
        self.neural_gpu_client: Optional[redis.Redis] = None
        self.engine_logic_client: Optional[redis.Redis] = None
        
        # Discovery settings
        self.heartbeat_interval = 30  # seconds
        self.discovery_channel = "nautilus:engine_discovery"
        self.partnership_channel = "nautilus:engine_partnerships"
        
        # Event handlers
        self.event_handlers: Dict[DiscoveryEventType, Callable] = {}
        self.partnership_handlers: Dict[str, Callable] = {}
        
        # Running tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._discovery_listener_task: Optional[asyncio.Task] = None
        self._partnership_listener_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        self._running = False
    
    async def initialize(self):
        """Initialize discovery protocol"""
        try:
            env = get_nautilus_environment()
            
            # Connect to Neural-GPU Bus for discovery announcements
            neural_bus = env.get_messagebus_by_type(MessageBusType.NEURAL_GPU_BUS)
            self.neural_gpu_client = redis.Redis(
                host="localhost",
                port=neural_bus.port,
                db=0,
                decode_responses=True
            )
            
            # Connect to Engine Logic Bus for partnerships
            logic_bus = env.get_messagebus_by_type(MessageBusType.ENGINE_LOGIC_BUS)
            self.engine_logic_client = redis.Redis(
                host="localhost",
                port=logic_bus.port,
                db=0,
                decode_responses=True
            )
            
            # Test connections
            await self.neural_gpu_client.ping()
            await self.engine_logic_client.ping()
            
            logger.info(f"Discovery protocol initialized for {self.identity.engine_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize discovery protocol: {e}")
            raise
    
    async def start(self):
        """Start discovery protocol"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._discovery_listener_task = asyncio.create_task(self._discovery_listener())
        self._partnership_listener_task = asyncio.create_task(self._partnership_listener())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Announce this engine
        await self.announce_engine()
        
        logger.info(f"Discovery protocol started for {self.identity.engine_id}")
    
    async def stop(self):
        """Stop discovery protocol"""
        if not self._running:
            return
        
        self._running = False
        
        # Send shutdown announcement
        await self.announce_shutdown()
        
        # Cancel tasks
        for task in [self._heartbeat_task, self._discovery_listener_task, 
                    self._partnership_listener_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close connections
        if self.neural_gpu_client:
            await self.neural_gpu_client.close()
        if self.engine_logic_client:
            await self.engine_logic_client.close()
        
        logger.info(f"Discovery protocol stopped for {self.identity.engine_id}")
    
    async def announce_engine(self):
        """Announce this engine to the network"""
        announcement = self.identity.to_discovery_announcement()
        
        message = DiscoveryMessage(
            event_type=DiscoveryEventType.ENGINE_ANNOUNCEMENT,
            source_engine_id=self.identity.engine_id,
            payload=announcement
        )
        
        await self._publish_discovery_message(message)
        logger.info(f"Announced engine {self.identity.engine_id}")
    
    async def announce_shutdown(self):
        """Announce engine shutdown"""
        message = DiscoveryMessage(
            event_type=DiscoveryEventType.ENGINE_SHUTDOWN,
            source_engine_id=self.identity.engine_id,
            payload={"reason": "graceful_shutdown"}
        )
        
        await self._publish_discovery_message(message)
        logger.info(f"Announced shutdown for {self.identity.engine_id}")
    
    async def propose_partnership(self, proposal: PartnershipProposal) -> bool:
        """Propose partnership with another engine"""
        message = DiscoveryMessage(
            event_type=DiscoveryEventType.PARTNERSHIP_REQUEST,
            source_engine_id=self.identity.engine_id,
            target_engine_id=proposal.target_engine_id,
            payload=asdict(proposal)
        )
        
        await self._publish_partnership_message(message)
        logger.info(f"Sent partnership proposal to {proposal.target_engine_id}")
        return True
    
    async def respond_to_partnership(self, proposal_id: str, accepted: bool, reason: str = ""):
        """Respond to partnership proposal"""
        # Implementation for partnership response
        pass
    
    def register_event_handler(self, event_type: DiscoveryEventType, handler: Callable):
        """Register handler for discovery events"""
        self.event_handlers[event_type] = handler
    
    def get_discovered_engines(self) -> Dict[str, Dict[str, Any]]:
        """Get all discovered engines"""
        return self.registry.get_all_engines()
    
    def get_online_engines(self) -> Dict[str, Dict[str, Any]]:
        """Get online engines only"""
        return self.registry.get_online_engines()
    
    def find_compatible_partners(self) -> List[str]:
        """Find engines compatible with this one"""
        compatible = []
        
        for engine_id, engine_data in self.registry.get_online_engines().items():
            if engine_id == self.identity.engine_id:
                continue
            
            # Check if engine is in preferred partners
            if engine_id in self.identity.get_preferred_partners():
                compatible.append(engine_id)
                continue
            
            # Check capability compatibility
            their_capabilities = set(engine_data.get("capabilities", []))
            my_capabilities = set([cap.value for cap in self.identity.capabilities.processing_capabilities])
            
            # Look for complementary capabilities
            if len(their_capabilities.intersection(my_capabilities)) > 0:
                compatible.append(engine_id)
        
        return compatible
    
    async def auto_establish_partnerships(self):
        """Automatically establish partnerships with compatible engines"""
        compatible_partners = self.find_compatible_partners()
        
        for partner_id in compatible_partners:
            # Check if partnership already exists
            existing_partnerships = self.registry.partnerships.get(self.identity.engine_id, [])
            if any(p["partner_id"] == partner_id for p in existing_partnerships):
                continue
            
            # Create partnership proposal
            proposal = PartnershipProposal(
                proposer_engine_id=self.identity.engine_id,
                target_engine_id=partner_id,
                relationship_type="secondary",
                data_flow_direction="bidirectional",
                proposed_message_types=["GENERAL_COMMUNICATION"],
                expected_latency_ms=10.0,
                expected_throughput=1000,
                reliability_requirement=99.0,
                benefits_description="Automatic partnership based on compatibility"
            )
            
            await self.propose_partnership(proposal)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self._running:
            try:
                # Update engine health
                self.identity.update_health_status()
                
                # Send heartbeat
                message = DiscoveryMessage(
                    event_type=DiscoveryEventType.ENGINE_HEARTBEAT,
                    source_engine_id=self.identity.engine_id,
                    payload={
                        "health": asdict(self.identity.health),
                        "performance": asdict(self.identity.performance_metrics)
                    }
                )
                
                await self._publish_discovery_message(message)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _discovery_listener(self):
        """Listen for discovery messages"""
        try:
            pubsub = self.neural_gpu_client.pubsub()
            await pubsub.subscribe(self.discovery_channel)
            
            while self._running:
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        await self._handle_discovery_message(message['data'])
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing discovery message: {e}")
                    
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await pubsub.unsubscribe(self.discovery_channel)
            except:
                pass
    
    async def _partnership_listener(self):
        """Listen for partnership messages"""
        try:
            pubsub = self.engine_logic_client.pubsub()
            await pubsub.subscribe(self.partnership_channel)
            
            while self._running:
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        await self._handle_partnership_message(message['data'])
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing partnership message: {e}")
                    
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await pubsub.unsubscribe(self.partnership_channel)
            except:
                pass
    
    async def _cleanup_loop(self):
        """Cleanup stale engines periodically"""
        while self._running:
            try:
                stale_engines = self.registry.check_stale_engines(timeout_seconds=90)
                if stale_engines:
                    logger.info(f"Marked engines as offline: {stale_engines}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(10)
    
    async def _publish_discovery_message(self, message: DiscoveryMessage):
        """Publish discovery message"""
        try:
            message_json = json.dumps(asdict(message), default=str)
            await self.neural_gpu_client.publish(self.discovery_channel, message_json)
        except Exception as e:
            logger.error(f"Failed to publish discovery message: {e}")
    
    async def _publish_partnership_message(self, message: DiscoveryMessage):
        """Publish partnership message"""
        try:
            message_json = json.dumps(asdict(message), default=str)
            await self.engine_logic_client.publish(self.partnership_channel, message_json)
        except Exception as e:
            logger.error(f"Failed to publish partnership message: {e}")
    
    async def _handle_discovery_message(self, message_data: str):
        """Handle incoming discovery message"""
        try:
            data = json.loads(message_data)
            event_type = DiscoveryEventType(data["event_type"])
            source_engine = data["source_engine_id"]
            
            # Ignore our own messages
            if source_engine == self.identity.engine_id:
                return
            
            if event_type == DiscoveryEventType.ENGINE_ANNOUNCEMENT:
                self.registry.register_engine(data["payload"])
                logger.info(f"Discovered new engine: {source_engine}")
                
                # Auto-establish partnerships if compatible
                await self.auto_establish_partnerships()
                
            elif event_type == DiscoveryEventType.ENGINE_HEARTBEAT:
                self.registry.update_heartbeat(source_engine)
                
                # Update engine data if provided
                if "payload" in data:
                    engine_data = self.registry.get_engine(source_engine)
                    if engine_data:
                        engine_data.update(data["payload"])
                        
            elif event_type == DiscoveryEventType.ENGINE_SHUTDOWN:
                self.registry.mark_engine_offline(source_engine)
                logger.info(f"Engine {source_engine} announced shutdown")
            
            # Call custom event handler if registered
            if event_type in self.event_handlers:
                await self.event_handlers[event_type](data)
                
        except Exception as e:
            logger.error(f"Error handling discovery message: {e}")
    
    async def _handle_partnership_message(self, message_data: str):
        """Handle incoming partnership message"""
        try:
            data = json.loads(message_data)
            event_type = DiscoveryEventType(data["event_type"])
            source_engine = data["source_engine_id"]
            target_engine = data.get("target_engine_id")
            
            # Only process messages targeted at us
            if target_engine and target_engine != self.identity.engine_id:
                return
            
            if event_type == DiscoveryEventType.PARTNERSHIP_REQUEST:
                await self._handle_partnership_request(data)
            elif event_type == DiscoveryEventType.PARTNERSHIP_RESPONSE:
                await self._handle_partnership_response(data)
            
        except Exception as e:
            logger.error(f"Error handling partnership message: {e}")
    
    async def _handle_partnership_request(self, message_data: Dict[str, Any]):
        """Handle partnership request"""
        proposal_data = message_data["payload"]
        proposer_id = message_data["source_engine_id"]
        
        # For now, auto-accept partnerships from preferred partners
        preferred_partners = self.identity.get_preferred_partners()
        
        if proposer_id in preferred_partners:
            # Auto-accept
            self.registry.add_partnership(
                self.identity.engine_id,
                proposer_id,
                {
                    "relationship_type": proposal_data.get("relationship_type", "secondary"),
                    "proposal_id": proposal_data.get("proposal_id"),
                    "status": "accepted",
                    "auto_accepted": True
                }
            )
            
            # Record in engine identity
            self.identity.add_partnership(proposer_id, proposal_data)
            
            logger.info(f"Auto-accepted partnership with {proposer_id}")
        
        # TODO: Implement more sophisticated partnership evaluation
    
    async def _handle_partnership_response(self, message_data: Dict[str, Any]):
        """Handle partnership response"""
        # TODO: Implement partnership response handling
        pass


if __name__ == "__main__":
    # Demo usage
    from .engine_identity import create_ml_engine_identity
    
    async def demo():
        ml_engine = create_ml_engine_identity()
        discovery = EngineDiscoveryProtocol(ml_engine)
        
        try:
            await discovery.initialize()
            await discovery.start()
            
            # Let it run for a bit
            await asyncio.sleep(10)
            
            # Check discovered engines
            engines = discovery.get_online_engines()
            print(f"Discovered {len(engines)} engines:")
            for engine_id in engines:
                print(f"  - {engine_id}")
            
        finally:
            await discovery.stop()
    
    asyncio.run(demo())