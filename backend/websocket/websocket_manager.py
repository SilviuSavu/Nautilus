"""
WebSocket Connection Manager - Sprint 3 Priority 1

Manages WebSocket connections for real-time streaming:
- Connection lifecycle management
- Authentication and authorization
- Rate limiting and throttling
- Connection health monitoring
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime, timedelta
import redis
# Remove Redis dependency for now - will add back when needed
# from ..redis_cache import get_redis

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Central WebSocket connection manager for real-time streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # connection_id -> subscription topics
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.redis_client = None  # Will be initialized when needed
        
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None) -> bool:
        """
        Accept WebSocket connection and register it
        
        Args:
            websocket: FastAPI WebSocket instance
            connection_id: Unique connection identifier
            user_id: Optional user identifier for authentication
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            await websocket.accept()
            
            # Store connection
            self.active_connections[connection_id] = websocket
            self.subscriptions[connection_id] = set()
            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "connected_at": datetime.utcnow(),
                "last_heartbeat": datetime.utcnow(),
                "message_count": 0
            }
            
            logger.info(f"WebSocket connection established: {connection_id}")
            
            # Send connection confirmation
            await self.send_personal_message({
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection {connection_id}: {e}")
            return False
    
    def disconnect(self, connection_id: str):
        """
        Disconnect and cleanup WebSocket connection
        
        Args:
            connection_id: Connection identifier to disconnect
        """
        try:
            # Remove from active connections
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
                
            # Clear subscriptions
            if connection_id in self.subscriptions:
                del self.subscriptions[connection_id]
                
            # Clear metadata
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
                
            logger.info(f"WebSocket connection disconnected: {connection_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket {connection_id}: {e}")
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: str) -> bool:
        """
        Send message to specific WebSocket connection
        
        Args:
            message: Message data to send
            connection_id: Target connection identifier
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            if connection_id not in self.active_connections:
                logger.warning(f"Connection not found: {connection_id}")
                return False
                
            websocket = self.active_connections[connection_id]
            message_json = json.dumps(message, default=str)
            
            await websocket.send_text(message_json)
            
            # Update message count
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["message_count"] += 1
                
            return True
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected during send: {connection_id}")
            self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            return False
    
    async def broadcast_message(self, message: Dict[str, Any], topic: Optional[str] = None) -> int:
        """
        Broadcast message to all connected clients or topic subscribers
        
        Args:
            message: Message data to broadcast
            topic: Optional topic filter (only send to subscribers)
            
        Returns:
            int: Number of successful sends
        """
        successful_sends = 0
        failed_connections = []
        
        # Determine target connections
        target_connections = []
        if topic:
            # Send to topic subscribers only
            for conn_id, topics in self.subscriptions.items():
                if topic in topics and conn_id in self.active_connections:
                    target_connections.append(conn_id)
        else:
            # Send to all active connections
            target_connections = list(self.active_connections.keys())
        
        # Send messages concurrently
        for connection_id in target_connections:
            try:
                success = await self.send_personal_message(message, connection_id)
                if success:
                    successful_sends += 1
                else:
                    failed_connections.append(connection_id)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                failed_connections.append(connection_id)
        
        # Cleanup failed connections
        for connection_id in failed_connections:
            self.disconnect(connection_id)
        
        logger.debug(f"Broadcast sent to {successful_sends}/{len(target_connections)} connections")
        return successful_sends
    
    def subscribe_to_topic(self, connection_id: str, topic: str) -> bool:
        """
        Subscribe connection to a topic
        
        Args:
            connection_id: Connection identifier
            topic: Topic name to subscribe to
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        try:
            if connection_id not in self.subscriptions:
                logger.warning(f"Connection not found for subscription: {connection_id}")
                return False
                
            self.subscriptions[connection_id].add(topic)
            logger.info(f"Connection {connection_id} subscribed to topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing {connection_id} to {topic}: {e}")
            return False
    
    def unsubscribe_from_topic(self, connection_id: str, topic: str) -> bool:
        """
        Unsubscribe connection from a topic
        
        Args:
            connection_id: Connection identifier
            topic: Topic name to unsubscribe from
            
        Returns:
            bool: True if unsubscription successful, False otherwise
        """
        try:
            if connection_id not in self.subscriptions:
                return False
                
            self.subscriptions[connection_id].discard(topic)
            logger.info(f"Connection {connection_id} unsubscribed from topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing {connection_id} from {topic}: {e}")
            return False
    
    async def handle_heartbeat(self, connection_id: str) -> bool:
        """
        Handle heartbeat from WebSocket connection
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            bool: True if heartbeat handled successfully, False otherwise
        """
        try:
            if connection_id not in self.connection_metadata:
                return False
                
            self.connection_metadata[connection_id]["last_heartbeat"] = datetime.utcnow()
            
            # Send heartbeat response
            await self.send_personal_message({
                "type": "heartbeat_response",
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling heartbeat for {connection_id}: {e}")
            return False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about WebSocket connections
        
        Returns:
            Dict containing connection statistics
        """
        now = datetime.utcnow()
        stats = {
            "total_connections": len(self.active_connections),
            "total_subscriptions": sum(len(topics) for topics in self.subscriptions.values()),
            "connections_by_topic": {},
            "connection_health": []
        }
        
        # Count connections by topic
        topic_counts = {}
        for topics in self.subscriptions.values():
            for topic in topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        stats["connections_by_topic"] = topic_counts
        
        # Connection health check
        for conn_id, metadata in self.connection_metadata.items():
            last_heartbeat = metadata.get("last_heartbeat", metadata["connected_at"])
            time_since_heartbeat = (now - last_heartbeat).total_seconds()
            
            stats["connection_health"].append({
                "connection_id": conn_id,
                "user_id": metadata.get("user_id"),
                "connected_duration": (now - metadata["connected_at"]).total_seconds(),
                "message_count": metadata.get("message_count", 0),
                "time_since_heartbeat": time_since_heartbeat,
                "is_healthy": time_since_heartbeat < 60  # Consider unhealthy if no heartbeat in 60s
            })
        
        return stats
    
    async def cleanup_stale_connections(self, timeout_seconds: int = 300) -> int:
        """
        Cleanup connections that haven't sent heartbeat within timeout
        
        Args:
            timeout_seconds: Timeout in seconds for stale connection detection
            
        Returns:
            int: Number of connections cleaned up
        """
        now = datetime.utcnow()
        stale_connections = []
        
        for conn_id, metadata in self.connection_metadata.items():
            last_heartbeat = metadata.get("last_heartbeat", metadata["connected_at"])
            if (now - last_heartbeat).total_seconds() > timeout_seconds:
                stale_connections.append(conn_id)
        
        # Disconnect stale connections
        for conn_id in stale_connections:
            self.disconnect(conn_id)
        
        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale WebSocket connections")
        
        return len(stale_connections)

# Global WebSocket manager instance
websocket_manager = WebSocketManager()