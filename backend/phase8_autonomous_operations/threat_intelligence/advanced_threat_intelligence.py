"""
Advanced Threat Intelligence System
Behavioral Analysis & Intelligence Aggregation for Financial Trading Platforms

Provides comprehensive threat intelligence with behavioral modeling, external feed
integration, and predictive threat analysis for proactive security operations.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import uuid
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
import aiohttp
from urllib.parse import urlparse
import re

logger = logging.getLogger(__name__)


class ThreatIntelligenceSource(Enum):
    """External threat intelligence sources"""
    INTERNAL_HONEYPOT = "internal_honeypot"
    MITRE_ATT_CK = "mitre_attack"
    STIX_TAXII = "stix_taxii"
    COMMERCIAL_FEEDS = "commercial_feeds"
    OSINT = "open_source_intelligence"
    DARK_WEB_MONITORING = "dark_web_monitoring"
    GOVERNMENT_FEEDS = "government_feeds"
    INDUSTRY_SHARING = "industry_sharing"
    THREAT_HUNTING = "threat_hunting"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"


class IndicatorType(Enum):
    """Types of threat indicators"""
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH = "file_hash"
    EMAIL = "email"
    USER_AGENT = "user_agent"
    PATTERN = "pattern"
    BEHAVIOR = "behavior"
    CRYPTOCURRENCY_ADDRESS = "crypto_address"
    API_KEY = "api_key"


class ThreatActorProfile(Enum):
    """Known threat actor categories"""
    NATION_STATE = "nation_state"
    CYBERCRIMINAL = "cybercriminal"
    HACKTIVIST = "hacktivist"
    INSIDER_THREAT = "insider_threat"
    SCRIPT_KIDDIE = "script_kiddie"
    ADVANCED_PERSISTENT_THREAT = "apt"
    FINANCIAL_CRIMINAL = "financial_criminal"
    MARKET_MANIPULATOR = "market_manipulator"


@dataclass
class ThreatIndicator:
    """Threat indicator with intelligence context"""
    indicator_id: str
    indicator_type: IndicatorType
    value: str
    confidence: float  # 0.0-1.0
    severity: str
    first_seen: datetime
    last_seen: datetime
    source: ThreatIntelligenceSource
    threat_actor: Optional[ThreatActorProfile] = None
    kill_chain_phase: Optional[str] = None
    mitre_techniques: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    false_positive_likelihood: float = 0.0
    expiry_date: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class BehavioralProfile:
    """User/System behavioral profile for anomaly detection"""
    profile_id: str
    entity_type: str  # user, system, api_key, etc.
    entity_id: str
    baseline_established: datetime
    activity_patterns: Dict[str, Any]
    risk_indicators: Dict[str, float]
    anomaly_thresholds: Dict[str, float]
    last_updated: datetime
    confidence_level: float = 0.8
    learning_samples: int = 0
    behavioral_fingerprint: str = ""


@dataclass
class ThreatCampaign:
    """Coordinated threat campaign tracking"""
    campaign_id: str
    name: str
    threat_actor: ThreatActorProfile
    start_date: datetime
    end_date: Optional[datetime]
    target_sectors: List[str]
    tactics: List[str]  # MITRE ATT&CK tactics
    techniques: List[str]  # MITRE ATT&CK techniques
    indicators: List[str]  # Associated indicator IDs
    confidence: float
    impact_assessment: Dict[str, Any]
    countermeasures: List[str]
    attribution_evidence: List[Dict[str, Any]]
    active: bool = True


class BehavioralAnalyzer:
    """Advanced behavioral analysis engine"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.behavioral_profiles = {}
        self.anomaly_models = {}
        self.learning_rate = 0.1
        
    async def analyze_behavior(self, entity_id: str, entity_type: str, 
                             activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze entity behavior and detect anomalies
        """
        try:
            # Get or create behavioral profile
            profile = await self._get_behavioral_profile(entity_id, entity_type)
            
            # Extract behavioral features
            features = self._extract_behavioral_features(activity_data)
            
            # Update profile with new data
            await self._update_behavioral_profile(profile, features)
            
            # Detect anomalies
            anomalies = await self._detect_behavioral_anomalies(profile, features)
            
            # Calculate risk score
            risk_score = self._calculate_behavioral_risk(profile, anomalies)
            
            return {
                "entity_id": entity_id,
                "risk_score": risk_score,
                "anomalies": anomalies,
                "behavioral_changes": self._detect_behavioral_changes(profile, features),
                "confidence": profile.confidence_level,
                "recommendations": self._generate_behavior_recommendations(anomalies)
            }
            
        except Exception as e:
            logger.error(f"Behavioral analysis failed for {entity_id}: {e}")
            return {"error": str(e)}
    
    async def _get_behavioral_profile(self, entity_id: str, 
                                    entity_type: str) -> BehavioralProfile:
        """Get or create behavioral profile"""
        profile_key = f"behavioral_profile:{entity_type}:{entity_id}"
        
        # Try to get existing profile
        profile_data = await self.redis.hgetall(profile_key)
        
        if profile_data:
            return BehavioralProfile(**profile_data)
        
        # Create new profile
        profile = BehavioralProfile(
            profile_id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            baseline_established=datetime.now(),
            activity_patterns={},
            risk_indicators={},
            anomaly_thresholds=self._get_default_thresholds(entity_type),
            last_updated=datetime.now()
        )
        
        # Store profile
        await self.redis.hset(profile_key, mapping=profile.__dict__)
        await self.redis.expire(profile_key, 86400 * 30)  # 30 days
        
        return profile
    
    def _extract_behavioral_features(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract behavioral features from activity data"""
        features = {
            "timestamp": datetime.now(),
            "activity_volume": activity_data.get("activity_count", 0),
            "data_transfer": activity_data.get("bytes_transferred", 0),
            "error_rate": activity_data.get("error_rate", 0.0),
            "response_times": activity_data.get("response_times", []),
            "unique_resources": len(set(activity_data.get("resources", []))),
            "geographic_locations": set(activity_data.get("locations", [])),
            "time_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "session_duration": activity_data.get("session_duration", 0),
            "concurrent_sessions": activity_data.get("concurrent_sessions", 1)
        }
        
        # Trading-specific features
        if "trading_activity" in activity_data:
            trading = activity_data["trading_activity"]
            features.update({
                "order_frequency": trading.get("order_count", 0),
                "position_size_variance": np.var(trading.get("position_sizes", [0])),
                "instrument_diversity": len(set(trading.get("instruments", []))),
                "pnl_volatility": np.std(trading.get("pnl_history", [0])),
                "risk_exposure": trading.get("current_exposure", 0)
            })
        
        return features
    
    async def _detect_behavioral_anomalies(self, profile: BehavioralProfile, 
                                         features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies using statistical methods"""
        anomalies = []
        
        for feature_name, current_value in features.items():
            if not isinstance(current_value, (int, float)):
                continue
                
            # Get historical baseline
            baseline = profile.activity_patterns.get(feature_name, {})
            if not baseline:
                continue
            
            mean = baseline.get("mean", current_value)
            std = baseline.get("std", 0)
            
            if std > 0:
                z_score = abs(current_value - mean) / std
                threshold = profile.anomaly_thresholds.get(feature_name, 3.0)
                
                if z_score > threshold:
                    anomalies.append({
                        "feature": feature_name,
                        "current_value": current_value,
                        "baseline_mean": mean,
                        "z_score": z_score,
                        "severity": "high" if z_score > 5.0 else "medium",
                        "anomaly_type": "statistical_deviation"
                    })
        
        return anomalies
    
    def _calculate_behavioral_risk(self, profile: BehavioralProfile, 
                                 anomalies: List[Dict[str, Any]]) -> float:
        """Calculate overall behavioral risk score"""
        base_risk = 0.1  # Base risk level
        
        # Anomaly contribution
        anomaly_risk = min(0.8, len(anomalies) * 0.1)
        
        # High severity anomalies have more weight
        high_severity_count = sum(1 for a in anomalies if a.get("severity") == "high")
        severity_risk = min(0.5, high_severity_count * 0.2)
        
        # Profile maturity affects confidence
        maturity_factor = min(1.0, profile.learning_samples / 1000)
        
        total_risk = (base_risk + anomaly_risk + severity_risk) * maturity_factor
        return min(1.0, total_risk)


class ThreatIntelligenceFeed:
    """External threat intelligence feed integration"""
    
    def __init__(self, source: ThreatIntelligenceSource, config: Dict[str, Any]):
        self.source = source
        self.config = config
        self.last_updated = None
        self.indicators_collected = 0
        self.error_count = 0
        
    async def fetch_indicators(self) -> List[ThreatIndicator]:
        """Fetch threat indicators from external source"""
        try:
            if self.source == ThreatIntelligenceSource.MITRE_ATT_CK:
                return await self._fetch_mitre_indicators()
            elif self.source == ThreatIntelligenceSource.OSINT:
                return await self._fetch_osint_indicators()
            elif self.source == ThreatIntelligenceSource.COMMERCIAL_FEEDS:
                return await self._fetch_commercial_indicators()
            else:
                logger.warning(f"No handler for source: {self.source}")
                return []
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to fetch from {self.source}: {e}")
            return []
    
    async def _fetch_mitre_indicators(self) -> List[ThreatIndicator]:
        """Fetch indicators from MITRE ATT&CK framework"""
        indicators = []
        
        # This would fetch from actual MITRE ATT&CK API
        # Placeholder implementation
        mitre_techniques = [
            {"id": "T1078", "name": "Valid Accounts", "tactics": ["initial-access"]},
            {"id": "T1190", "name": "Exploit Public-Facing Application", "tactics": ["initial-access"]},
            {"id": "T1566", "name": "Phishing", "tactics": ["initial-access"]}
        ]
        
        for technique in mitre_techniques:
            indicator = ThreatIndicator(
                indicator_id=f"mitre_{technique['id']}",
                indicator_type=IndicatorType.PATTERN,
                value=technique["name"],
                confidence=0.9,
                severity="high",
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                source=self.source,
                mitre_techniques=[technique["id"]],
                context=technique
            )
            indicators.append(indicator)
        
        return indicators
    
    async def _fetch_osint_indicators(self) -> List[ThreatIndicator]:
        """Fetch indicators from open source intelligence"""
        indicators = []
        
        # Example OSINT sources (would use real APIs)
        osint_sources = [
            "https://api.threatfox.abuse.ch/api/v1/",
            "https://urlhaus-api.abuse.ch/v1/",
            "https://api.malwarebazaar.abuse.ch/api/v1/"
        ]
        
        # Placeholder for actual OSINT collection
        for source_url in osint_sources:
            # This would make actual API calls
            pass
        
        return indicators


class ThreatCampaignTracker:
    """Track coordinated threat campaigns"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.active_campaigns = {}
        self.campaign_patterns = {}
        
    async def analyze_campaign_activity(self, indicators: List[ThreatIndicator]) -> List[ThreatCampaign]:
        """Analyze indicators for coordinated campaign activity"""
        campaigns = []
        
        # Group indicators by similarity
        indicator_groups = self._cluster_indicators(indicators)
        
        for group in indicator_groups:
            # Check if group matches known campaign pattern
            campaign = await self._match_campaign_pattern(group)
            
            if campaign:
                campaigns.append(campaign)
            elif len(group) >= 3:  # New potential campaign
                new_campaign = await self._create_campaign(group)
                if new_campaign:
                    campaigns.append(new_campaign)
        
        return campaigns
    
    def _cluster_indicators(self, indicators: List[ThreatIndicator]) -> List[List[ThreatIndicator]]:
        """Cluster indicators by similarity and timing"""
        clusters = []
        
        # Simple clustering by time window and attributes
        time_window = timedelta(hours=24)
        
        for indicator in indicators:
            placed = False
            
            for cluster in clusters:
                # Check if indicator fits in existing cluster
                if self._indicators_similar(indicator, cluster[0], time_window):
                    cluster.append(indicator)
                    placed = True
                    break
            
            if not placed:
                clusters.append([indicator])
        
        return clusters
    
    def _indicators_similar(self, ind1: ThreatIndicator, ind2: ThreatIndicator, 
                          time_window: timedelta) -> bool:
        """Check if two indicators are similar enough to cluster"""
        # Time proximity
        if abs(ind1.first_seen - ind2.first_seen) > time_window:
            return False
        
        # Source similarity
        if ind1.source == ind2.source:
            return True
        
        # MITRE technique overlap
        if set(ind1.mitre_techniques) & set(ind2.mitre_techniques):
            return True
        
        # Threat actor similarity
        if ind1.threat_actor and ind2.threat_actor:
            return ind1.threat_actor == ind2.threat_actor
        
        return False


class AdvancedThreatIntelligence:
    """
    Main threat intelligence system coordinating all components
    """
    
    def __init__(self, redis_url: str, database_url: str):
        self.redis_url = redis_url
        self.database_url = database_url
        self.redis = None
        self.behavioral_analyzer = None
        self.campaign_tracker = None
        self.intelligence_feeds = []
        self.threat_indicators = {}
        self.active_campaigns = {}
        self.intelligence_metrics = defaultdict(int)
        
    async def initialize(self):
        """Initialize threat intelligence system"""
        try:
            # Initialize Redis connection
            self.redis = await redis.from_url(self.redis_url)
            
            # Initialize components
            self.behavioral_analyzer = BehavioralAnalyzer(self.redis)
            self.campaign_tracker = ThreatCampaignTracker(self.redis)
            
            # Initialize threat intelligence feeds
            await self._initialize_feeds()
            
            logger.info("Advanced Threat Intelligence system initialized")
            
        except Exception as e:
            logger.error(f"Threat Intelligence initialization failed: {e}")
            raise
    
    async def _initialize_feeds(self):
        """Initialize external threat intelligence feeds"""
        feed_configs = [
            {
                "source": ThreatIntelligenceSource.MITRE_ATT_CK,
                "config": {"api_url": "https://attack.mitre.org/api/"}
            },
            {
                "source": ThreatIntelligenceSource.OSINT,
                "config": {"sources": ["threatfox", "urlhaus", "malwarebazaar"]}
            }
        ]
        
        for feed_config in feed_configs:
            feed = ThreatIntelligenceFeed(
                feed_config["source"],
                feed_config["config"]
            )
            self.intelligence_feeds.append(feed)
        
        logger.info(f"Initialized {len(self.intelligence_feeds)} threat intelligence feeds")
    
    async def start_intelligence_collection(self):
        """Start threat intelligence collection from all sources"""
        try:
            # Start feed collection tasks
            collection_tasks = []
            for feed in self.intelligence_feeds:
                task = asyncio.create_task(self._collect_from_feed(feed))
                collection_tasks.append(task)
            
            # Start behavioral analysis task
            behavioral_task = asyncio.create_task(self._continuous_behavioral_analysis())
            collection_tasks.append(behavioral_task)
            
            # Start campaign tracking task
            campaign_task = asyncio.create_task(self._continuous_campaign_tracking())
            collection_tasks.append(campaign_task)
            
            logger.info("Threat intelligence collection started")
            
            # Wait for all tasks (in production, these would run indefinitely)
            await asyncio.gather(*collection_tasks)
            
        except Exception as e:
            logger.error(f"Intelligence collection failed: {e}")
    
    async def _collect_from_feed(self, feed: ThreatIntelligenceFeed):
        """Continuously collect from a threat intelligence feed"""
        while True:
            try:
                indicators = await feed.fetch_indicators()
                
                for indicator in indicators:
                    await self._process_threat_indicator(indicator)
                
                feed.indicators_collected += len(indicators)
                feed.last_updated = datetime.now()
                
                self.intelligence_metrics[f"indicators_{feed.source.value}"] += len(indicators)
                
                # Wait before next collection
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Feed collection error for {feed.source}: {e}")
                await asyncio.sleep(300)  # 5 minutes backoff
    
    async def _process_threat_indicator(self, indicator: ThreatIndicator):
        """Process and store threat indicator"""
        try:
            # Store indicator
            indicator_key = f"threat_indicator:{indicator.indicator_id}"
            await self.redis.hset(indicator_key, mapping=indicator.__dict__)
            await self.redis.expire(indicator_key, 86400 * 30)  # 30 days
            
            # Add to lookup indexes
            await self._index_indicator(indicator)
            
            # Check for matches with active threats
            await self._check_indicator_matches(indicator)
            
        except Exception as e:
            logger.error(f"Indicator processing failed: {e}")
    
    async def _index_indicator(self, indicator: ThreatIndicator):
        """Index indicator for fast lookups"""
        # Index by type
        type_key = f"indicators_by_type:{indicator.indicator_type.value}"
        await self.redis.sadd(type_key, indicator.indicator_id)
        
        # Index by value hash (for exact matching)
        value_hash = hashlib.sha256(indicator.value.encode()).hexdigest()
        hash_key = f"indicators_by_hash:{value_hash}"
        await self.redis.set(hash_key, indicator.indicator_id)
        
        # Index by source
        source_key = f"indicators_by_source:{indicator.source.value}"
        await self.redis.sadd(source_key, indicator.indicator_id)
    
    async def lookup_threat_indicator(self, indicator_value: str, 
                                    indicator_type: IndicatorType) -> Optional[ThreatIndicator]:
        """Look up threat indicator by value"""
        try:
            value_hash = hashlib.sha256(indicator_value.encode()).hexdigest()
            hash_key = f"indicators_by_hash:{value_hash}"
            
            indicator_id = await self.redis.get(hash_key)
            if not indicator_id:
                return None
            
            indicator_key = f"threat_indicator:{indicator_id}"
            indicator_data = await self.redis.hgetall(indicator_key)
            
            if indicator_data:
                return ThreatIndicator(**indicator_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Indicator lookup failed: {e}")
            return None
    
    async def analyze_entity_behavior(self, entity_id: str, entity_type: str, 
                                    activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entity behavior for threats"""
        if not self.behavioral_analyzer:
            return {"error": "Behavioral analyzer not initialized"}
        
        return await self.behavioral_analyzer.analyze_behavior(
            entity_id, entity_type, activity_data
        )
    
    async def get_threat_intelligence_status(self) -> Dict[str, Any]:
        """Get current threat intelligence status"""
        try:
            status = {
                "active_feeds": len(self.intelligence_feeds),
                "total_indicators": len(self.threat_indicators),
                "active_campaigns": len(self.active_campaigns),
                "intelligence_metrics": dict(self.intelligence_metrics),
                "feed_status": []
            }
            
            for feed in self.intelligence_feeds:
                feed_status = {
                    "source": feed.source.value,
                    "last_updated": feed.last_updated.isoformat() if feed.last_updated else None,
                    "indicators_collected": feed.indicators_collected,
                    "error_count": feed.error_count
                }
                status["feed_status"].append(feed_status)
            
            return status
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}


# Global threat intelligence instance
threat_intelligence = None

async def get_threat_intelligence() -> AdvancedThreatIntelligence:
    """Get or create threat intelligence instance"""
    global threat_intelligence
    if not threat_intelligence:
        threat_intelligence = AdvancedThreatIntelligence(
            redis_url="redis://localhost:6379",
            database_url="postgresql://localhost:5432/nautilus"
        )
        await threat_intelligence.initialize()
    return threat_intelligence


async def lookup_indicator(value: str, indicator_type: str) -> Optional[Dict[str, Any]]:
    """
    Main entry point for threat indicator lookup
    """
    try:
        ti_system = await get_threat_intelligence()
        indicator = await ti_system.lookup_threat_indicator(
            value, 
            IndicatorType(indicator_type)
        )
        
        if indicator:
            return {
                "found": True,
                "indicator_id": indicator.indicator_id,
                "confidence": indicator.confidence,
                "severity": indicator.severity,
                "threat_actor": indicator.threat_actor.value if indicator.threat_actor else None,
                "source": indicator.source.value,
                "first_seen": indicator.first_seen.isoformat(),
                "context": indicator.context
            }
        
        return {"found": False}
        
    except Exception as e:
        logger.error(f"Indicator lookup failed: {e}")
        return {"error": str(e)}