"""
Quantum-Safe Migration Tools
============================

Comprehensive migration tools for transitioning from classical to quantum-safe
cryptography while maintaining system availability and security during the
migration process.

Key Features:
- Automated cryptographic algorithm migration
- Hybrid classical/post-quantum deployment
- Zero-downtime migration strategies
- Migration validation and rollback
- Performance impact assessment
- Compliance-aware migration planning
- Risk-based migration prioritization

"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

from .post_quantum_crypto import PostQuantumCrypto, PQAlgorithm
from .quantum_audit import QuantumResistantAuditTrail, AuditEventType


class MigrationStrategy(Enum):
    """Migration deployment strategies"""
    BIG_BANG = "big_bang"  # Complete migration at once
    PHASED = "phased"  # Gradual phase-by-phase migration
    BLUE_GREEN = "blue_green"  # Parallel environment switch
    CANARY = "canary"  # Small subset testing
    ROLLING = "rolling"  # Service-by-service rolling update
    HYBRID_COEXISTENCE = "hybrid_coexistence"  # Long-term hybrid operation


class MigrationPhase(Enum):
    """Migration phases"""
    ASSESSMENT = "assessment"
    PLANNING = "planning"
    PREPARATION = "preparation"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION_ROLLOUT = "production_rollout"
    VALIDATION = "validation"
    COMPLETION = "completion"
    ROLLBACK = "rollback"


class MigrationStatus(Enum):
    """Migration status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"


class SystemComponent(Enum):
    """System components for migration"""
    AUTHENTICATION = "authentication"
    KEY_EXCHANGE = "key_exchange"
    DIGITAL_SIGNATURES = "digital_signatures"
    DATA_ENCRYPTION = "data_encryption"
    COMMUNICATION_PROTOCOLS = "communication_protocols"
    AUDIT_TRAILS = "audit_trails"
    SESSION_MANAGEMENT = "session_management"
    API_SECURITY = "api_security"
    DATABASE_ENCRYPTION = "database_encryption"
    BACKUP_ENCRYPTION = "backup_encryption"


@dataclass
class CryptographicAsset:
    """Cryptographic asset requiring migration"""
    asset_id: str
    asset_name: str
    asset_type: str
    current_algorithm: str
    key_size: int
    usage_frequency: int
    criticality_level: int  # 1-5 scale
    dependent_systems: List[str]
    migration_complexity: int  # 1-5 scale
    business_impact: int  # 1-5 scale
    compliance_requirements: List[str]
    estimated_migration_time: timedelta
    quantum_vulnerability_score: float  # 0.0-1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationPlan:
    """Comprehensive migration plan"""
    plan_id: str
    plan_name: str
    strategy: MigrationStrategy
    target_completion_date: datetime
    phases: List[Dict[str, Any]]
    assets_to_migrate: List[str]  # Asset IDs
    dependencies: Dict[str, List[str]]
    risk_assessment: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    success_criteria: List[str]
    stakeholders: List[str]
    estimated_cost: float
    estimated_duration: timedelta
    created_by: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationExecution:
    """Migration execution tracking"""
    execution_id: str
    plan_id: str
    current_phase: MigrationPhase
    status: MigrationStatus
    start_time: datetime
    current_phase_start: datetime
    estimated_completion: Optional[datetime]
    progress_percentage: float
    completed_phases: List[MigrationPhase]
    failed_steps: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    validation_results: Dict[str, Any]
    rollback_ready: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumSafeMigrationManager:
    """
    Comprehensive migration manager for transitioning to quantum-safe cryptography.
    
    Orchestrates the migration from classical to post-quantum cryptography
    across all system components while maintaining security, availability,
    and performance requirements.
    """
    
    def __init__(self,
                 migration_data_path: str = "/app/migration",
                 parallel_executions: int = 3,
                 enable_auto_rollback: bool = True,
                 performance_threshold_degradation: float = 0.2):
        """
        Initialize quantum-safe migration manager.
        
        Args:
            migration_data_path: Path for migration data storage
            parallel_executions: Maximum parallel migration executions
            enable_auto_rollback: Enable automatic rollback on failures
            performance_threshold_degradation: Max acceptable performance degradation
        """
        self.migration_data_path = migration_data_path
        self.parallel_executions = parallel_executions
        self.enable_auto_rollback = enable_auto_rollback
        self.performance_threshold = performance_threshold_degradation
        
        self.logger = logging.getLogger("quantum_safe.migration")
        
        # Migration state
        self.cryptographic_assets: Dict[str, CryptographicAsset] = {}
        self.migration_plans: Dict[str, MigrationPlan] = {}
        self.active_executions: Dict[str, MigrationExecution] = {}
        
        # Migration templates and patterns
        self.migration_templates: Dict[str, Dict[str, Any]] = {}
        self.migration_patterns: Dict[str, Any] = {}
        
        # Performance tracking
        self.migration_metrics = {
            "total_assets": 0,
            "migrated_assets": 0,
            "failed_migrations": 0,
            "rollbacks_performed": 0,
            "average_migration_time": 0.0,
            "performance_impact": 0.0,
            "success_rate": 0.0
        }
        
        # Components
        self.pq_crypto = PostQuantumCrypto()
        self.audit_trail = QuantumResistantAuditTrail("migration-audit")
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.parallel_executions)
        self._shutdown_event = threading.Event()
        
        # Initialize migration system
        asyncio.create_task(self._initialize_migration_system())
    
    async def _initialize_migration_system(self):
        """Initialize the migration management system"""
        try:
            self.logger.info("Initializing quantum-safe migration system")
            
            # Load migration templates
            await self._load_migration_templates()
            
            # Discover cryptographic assets
            await self._discover_cryptographic_assets()
            
            # Load migration patterns
            await self._load_migration_patterns()
            
            # Start monitoring loops
            asyncio.create_task(self._migration_monitoring_loop())
            asyncio.create_task(self._metrics_collection_loop())
            
            self.logger.info("Migration system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize migration system: {str(e)}")
            raise
    
    async def _load_migration_templates(self):
        """Load migration templates for different scenarios"""
        
        # Authentication system migration template
        self.migration_templates["authentication_migration"] = {
            "name": "Authentication System Migration",
            "strategy": MigrationStrategy.PHASED,
            "phases": [
                {
                    "name": "credential_store_migration",
                    "description": "Migrate credential storage to post-quantum",
                    "estimated_duration": timedelta(hours=4),
                    "dependencies": [],
                    "rollback_plan": "restore_classical_credential_store",
                    "validation_steps": ["verify_credential_integrity", "test_authentication_flow"]
                },
                {
                    "name": "session_management_migration",
                    "description": "Migrate session tokens to quantum-safe",
                    "estimated_duration": timedelta(hours=2),
                    "dependencies": ["credential_store_migration"],
                    "rollback_plan": "restore_classical_sessions",
                    "validation_steps": ["verify_session_security", "test_session_lifecycle"]
                },
                {
                    "name": "mfa_migration",
                    "description": "Migrate multi-factor authentication",
                    "estimated_duration": timedelta(hours=6),
                    "dependencies": ["session_management_migration"],
                    "rollback_plan": "restore_classical_mfa",
                    "validation_steps": ["verify_mfa_security", "test_biometric_auth"]
                }
            ],
            "success_criteria": [
                "all_authentication_flows_operational",
                "performance_degradation_under_20_percent",
                "security_validation_passed"
            ],
            "estimated_total_duration": timedelta(hours=12)
        }
        
        # Key exchange migration template
        self.migration_templates["key_exchange_migration"] = {
            "name": "Key Exchange Migration",
            "strategy": MigrationStrategy.HYBRID_COEXISTENCE,
            "phases": [
                {
                    "name": "hybrid_key_exchange_deployment",
                    "description": "Deploy hybrid classical/PQ key exchange",
                    "estimated_duration": timedelta(hours=6),
                    "dependencies": [],
                    "rollback_plan": "disable_pq_key_exchange",
                    "validation_steps": ["verify_hybrid_compatibility", "test_key_exchange_security"]
                },
                {
                    "name": "client_migration",
                    "description": "Migrate clients to support PQ key exchange",
                    "estimated_duration": timedelta(days=14),
                    "dependencies": ["hybrid_key_exchange_deployment"],
                    "rollback_plan": "revert_client_updates",
                    "validation_steps": ["verify_client_compatibility", "test_key_exchange_performance"]
                },
                {
                    "name": "classical_deprecation",
                    "description": "Phase out classical key exchange",
                    "estimated_duration": timedelta(days=7),
                    "dependencies": ["client_migration"],
                    "rollback_plan": "re_enable_classical_key_exchange",
                    "validation_steps": ["verify_pq_only_operation", "monitor_error_rates"]
                }
            ],
            "success_criteria": [
                "all_key_exchanges_quantum_safe",
                "client_compatibility_maintained",
                "performance_within_acceptable_limits"
            ],
            "estimated_total_duration": timedelta(days=21)
        }
        
        # Digital signatures migration template
        self.migration_templates["digital_signatures_migration"] = {
            "name": "Digital Signatures Migration",
            "strategy": MigrationStrategy.BLUE_GREEN,
            "phases": [
                {
                    "name": "pq_signature_infrastructure",
                    "description": "Deploy post-quantum signature infrastructure",
                    "estimated_duration": timedelta(hours=8),
                    "dependencies": [],
                    "rollback_plan": "switch_back_to_classical_signatures",
                    "validation_steps": ["verify_pq_signature_generation", "test_signature_verification"]
                },
                {
                    "name": "dual_signature_period",
                    "description": "Generate both classical and PQ signatures",
                    "estimated_duration": timedelta(days=30),
                    "dependencies": ["pq_signature_infrastructure"],
                    "rollback_plan": "stop_pq_signature_generation",
                    "validation_steps": ["verify_dual_signatures", "monitor_signature_performance"]
                },
                {
                    "name": "pq_signature_switchover",
                    "description": "Switch to PQ signatures only",
                    "estimated_duration": timedelta(hours=4),
                    "dependencies": ["dual_signature_period"],
                    "rollback_plan": "revert_to_classical_signatures",
                    "validation_steps": ["verify_pq_signature_acceptance", "test_signature_chain_validity"]
                }
            ],
            "success_criteria": [
                "all_signatures_quantum_safe",
                "signature_verification_reliable",
                "performance_impact_acceptable"
            ],
            "estimated_total_duration": timedelta(days=31)
        }
        
        self.logger.info(f"Loaded {len(self.migration_templates)} migration templates")
    
    async def _discover_cryptographic_assets(self):
        """Discover and inventory cryptographic assets in the system"""
        
        # Simulated asset discovery - in practice would scan system configurations
        assets = [
            CryptographicAsset(
                asset_id="auth-rsa-2048",
                asset_name="Authentication RSA Keys",
                asset_type="authentication",
                current_algorithm="RSA-2048",
                key_size=2048,
                usage_frequency=10000,  # per day
                criticality_level=5,
                dependent_systems=["web_api", "mobile_app", "trading_engine"],
                migration_complexity=4,
                business_impact=5,
                compliance_requirements=["SOX", "PCI-DSS"],
                estimated_migration_time=timedelta(hours=12),
                quantum_vulnerability_score=0.9
            ),
            CryptographicAsset(
                asset_id="ssl-ecc-p256",
                asset_name="TLS/SSL ECC Certificates",
                asset_type="communication",
                current_algorithm="ECDSA-P256",
                key_size=256,
                usage_frequency=50000,  # per day
                criticality_level=5,
                dependent_systems=["all_web_services", "api_gateway"],
                migration_complexity=3,
                business_impact=5,
                compliance_requirements=["PCI-DSS", "GDPR"],
                estimated_migration_time=timedelta(hours=8),
                quantum_vulnerability_score=0.95
            ),
            CryptographicAsset(
                asset_id="data-encrypt-aes256",
                asset_name="Database AES Encryption",
                asset_type="data_encryption",
                current_algorithm="AES-256",
                key_size=256,
                usage_frequency=100000,  # per day
                criticality_level=4,
                dependent_systems=["primary_database", "backup_systems"],
                migration_complexity=2,
                business_impact=3,
                compliance_requirements=["SOX", "GDPR"],
                estimated_migration_time=timedelta(hours=4),
                quantum_vulnerability_score=0.3  # AES-256 more resistant
            ),
            CryptographicAsset(
                asset_id="jwt-rsa-2048",
                asset_name="JWT Token Signatures",
                asset_type="api_security",
                current_algorithm="RSA-2048",
                key_size=2048,
                usage_frequency=25000,  # per day
                criticality_level=4,
                dependent_systems=["api_services", "microservices"],
                migration_complexity=3,
                business_impact=4,
                compliance_requirements=["OAUTH2", "OIDC"],
                estimated_migration_time=timedelta(hours=6),
                quantum_vulnerability_score=0.9
            ),
            CryptographicAsset(
                asset_id="audit-signatures",
                asset_name="Audit Log Signatures",
                asset_type="audit_trails",
                current_algorithm="ECDSA-P384",
                key_size=384,
                usage_frequency=5000,  # per day
                criticality_level=5,
                dependent_systems=["audit_system", "compliance_reporting"],
                migration_complexity=2,
                business_impact=4,
                compliance_requirements=["SOX", "FINRA", "SEC"],
                estimated_migration_time=timedelta(hours=3),
                quantum_vulnerability_score=0.95
            )
        ]
        
        # Store discovered assets
        for asset in assets:
            self.cryptographic_assets[asset.asset_id] = asset
        
        # Update metrics
        self.migration_metrics["total_assets"] = len(self.cryptographic_assets)
        
        self.logger.info(f"Discovered {len(assets)} cryptographic assets")
    
    async def _load_migration_patterns(self):
        """Load migration patterns and best practices"""
        
        self.migration_patterns = {
            "risk_based_prioritization": {
                "high_priority_criteria": [
                    "quantum_vulnerability_score > 0.8",
                    "criticality_level >= 4",
                    "compliance_requirements includes regulatory"
                ],
                "medium_priority_criteria": [
                    "quantum_vulnerability_score > 0.5",
                    "business_impact >= 3"
                ],
                "low_priority_criteria": [
                    "quantum_vulnerability_score <= 0.5",
                    "business_impact < 3"
                ]
            },
            "dependency_management": {
                "parallel_migration_safe": [
                    "independent_systems",
                    "different_asset_types",
                    "non_shared_infrastructure"
                ],
                "sequential_required": [
                    "shared_key_infrastructure",
                    "dependent_services",
                    "certificate_chains"
                ]
            },
            "performance_considerations": {
                "high_performance_impact": [
                    "signature_generation",
                    "key_exchange_handshakes",
                    "certificate_validation"
                ],
                "medium_performance_impact": [
                    "symmetric_encryption",
                    "hash_functions",
                    "random_number_generation"
                ],
                "low_performance_impact": [
                    "key_storage",
                    "configuration_changes",
                    "logging_enhancements"
                ]
            }
        }
        
        self.logger.info("Loaded migration patterns and best practices")
    
    async def assess_migration_readiness(self) -> Dict[str, Any]:
        """Comprehensive migration readiness assessment"""
        
        assessment = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_readiness_score": 0.0,
            "asset_analysis": {},
            "risk_assessment": {},
            "dependency_analysis": {},
            "performance_impact": {},
            "compliance_considerations": {},
            "recommendations": []
        }
        
        try:
            # Analyze each asset
            total_readiness_score = 0.0
            high_risk_assets = []
            critical_dependencies = []
            
            for asset_id, asset in self.cryptographic_assets.items():
                asset_readiness = self._calculate_asset_readiness(asset)
                assessment["asset_analysis"][asset_id] = asset_readiness
                
                total_readiness_score += asset_readiness["readiness_score"]
                
                if asset_readiness["risk_level"] == "high":
                    high_risk_assets.append(asset_id)
                
                if asset.migration_complexity >= 4:
                    critical_dependencies.append(asset_id)
            
            # Calculate overall readiness
            if len(self.cryptographic_assets) > 0:
                assessment["overall_readiness_score"] = total_readiness_score / len(self.cryptographic_assets)
            
            # Risk assessment
            assessment["risk_assessment"] = {
                "high_risk_assets": len(high_risk_assets),
                "critical_dependencies": len(critical_dependencies),
                "quantum_vulnerable_assets": len([
                    a for a in self.cryptographic_assets.values() 
                    if a.quantum_vulnerability_score > 0.7
                ]),
                "business_critical_assets": len([
                    a for a in self.cryptographic_assets.values() 
                    if a.business_impact >= 4
                ])
            }
            
            # Dependency analysis
            dependency_graph = self._build_dependency_graph()
            assessment["dependency_analysis"] = {
                "total_dependencies": len(dependency_graph),
                "circular_dependencies": self._detect_circular_dependencies(dependency_graph),
                "migration_waves": self._calculate_migration_waves(dependency_graph)
            }
            
            # Performance impact assessment
            assessment["performance_impact"] = await self._assess_performance_impact()
            
            # Compliance considerations
            assessment["compliance_considerations"] = self._assess_compliance_requirements()
            
            # Generate recommendations
            assessment["recommendations"] = self._generate_migration_recommendations(assessment)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error in migration readiness assessment: {str(e)}")
            assessment["error"] = str(e)
            return assessment
    
    def _calculate_asset_readiness(self, asset: CryptographicAsset) -> Dict[str, Any]:
        """Calculate readiness score for a specific asset"""
        
        readiness_factors = {
            "quantum_vulnerability": asset.quantum_vulnerability_score * 0.3,
            "migration_complexity": (5 - asset.migration_complexity) / 5 * 0.2,
            "business_impact": asset.business_impact / 5 * 0.2,
            "dependency_complexity": (5 - len(asset.dependent_systems)) / 5 * 0.15,
            "compliance_requirements": min(len(asset.compliance_requirements) / 3, 1.0) * 0.15
        }
        
        readiness_score = sum(readiness_factors.values())
        
        # Determine risk level
        if asset.quantum_vulnerability_score > 0.8 and asset.business_impact >= 4:
            risk_level = "critical"
        elif asset.quantum_vulnerability_score > 0.6 and asset.business_impact >= 3:
            risk_level = "high"
        elif asset.quantum_vulnerability_score > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "readiness_score": readiness_score,
            "risk_level": risk_level,
            "readiness_factors": readiness_factors,
            "migration_urgency": self._calculate_migration_urgency(asset),
            "recommended_strategy": self._recommend_migration_strategy(asset)
        }
    
    def _calculate_migration_urgency(self, asset: CryptographicAsset) -> str:
        """Calculate migration urgency based on quantum threat timeline"""
        
        # Simplified urgency calculation based on quantum vulnerability and business impact
        urgency_score = (asset.quantum_vulnerability_score * 0.7 + 
                        asset.business_impact / 5 * 0.3)
        
        if urgency_score > 0.8:
            return "immediate"
        elif urgency_score > 0.6:
            return "high"
        elif urgency_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _recommend_migration_strategy(self, asset: CryptographicAsset) -> MigrationStrategy:
        """Recommend migration strategy for an asset"""
        
        if asset.business_impact >= 5 and len(asset.dependent_systems) > 3:
            return MigrationStrategy.BLUE_GREEN
        elif asset.migration_complexity >= 4:
            return MigrationStrategy.PHASED
        elif asset.usage_frequency > 50000:
            return MigrationStrategy.CANARY
        elif asset.business_impact <= 2:
            return MigrationStrategy.BIG_BANG
        else:
            return MigrationStrategy.ROLLING
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph between assets"""
        
        dependency_graph = {}
        
        for asset_id, asset in self.cryptographic_assets.items():
            dependencies = []
            
            # Find assets that this asset depends on
            for other_id, other_asset in self.cryptographic_assets.items():
                if other_id != asset_id:
                    # Check for system dependencies
                    common_systems = set(asset.dependent_systems) & set(other_asset.dependent_systems)
                    if common_systems:
                        # If assets share systems and one is foundational, create dependency
                        if (other_asset.asset_type in ["authentication", "key_exchange"] and 
                            asset.asset_type not in ["authentication", "key_exchange"]):
                            dependencies.append(other_id)
            
            dependency_graph[asset_id] = dependencies
        
        return dependency_graph
    
    def _detect_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies in the dependency graph"""
        
        def dfs(node: str, path: List[str], visited: Set[str]) -> List[List[str]]:
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                return [path[cycle_start:] + [node]]
            
            if node in visited:
                return []
            
            visited.add(node)
            path.append(node)
            
            cycles = []
            for dependency in dependency_graph.get(node, []):
                cycles.extend(dfs(dependency, path.copy(), visited))
            
            return cycles
        
        all_cycles = []
        visited = set()
        
        for node in dependency_graph:
            if node not in visited:
                cycles = dfs(node, [], visited)
                all_cycles.extend(cycles)
        
        return all_cycles
    
    def _calculate_migration_waves(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Calculate migration waves based on dependencies"""
        
        waves = []
        remaining_assets = set(dependency_graph.keys())
        
        while remaining_assets:
            current_wave = []
            
            # Find assets with no unresolved dependencies
            for asset_id in list(remaining_assets):
                dependencies = dependency_graph[asset_id]
                unresolved_dependencies = [dep for dep in dependencies if dep in remaining_assets]
                
                if not unresolved_dependencies:
                    current_wave.append(asset_id)
            
            if not current_wave:
                # Handle circular dependencies by including the least complex assets
                min_complexity = min(
                    self.cryptographic_assets[aid].migration_complexity 
                    for aid in remaining_assets
                )
                current_wave = [
                    aid for aid in remaining_assets 
                    if self.cryptographic_assets[aid].migration_complexity == min_complexity
                ][:1]  # Take one to break the cycle
            
            waves.append(current_wave)
            remaining_assets -= set(current_wave)
        
        return waves
    
    async def _assess_performance_impact(self) -> Dict[str, Any]:
        """Assess performance impact of migration"""
        
        impact_assessment = {
            "cpu_overhead_estimate": 0.0,
            "memory_overhead_estimate": 0.0,
            "network_overhead_estimate": 0.0,
            "latency_impact_estimate": 0.0,
            "throughput_impact_estimate": 0.0,
            "high_impact_operations": [],
            "mitigation_strategies": []
        }
        
        total_usage = sum(asset.usage_frequency for asset in self.cryptographic_assets.values())
        
        for asset in self.cryptographic_assets.values():
            usage_weight = asset.usage_frequency / max(total_usage, 1)
            
            # Estimate performance impact based on algorithm type
            if "RSA" in asset.current_algorithm:
                # RSA to post-quantum signature impact
                impact_assessment["cpu_overhead_estimate"] += usage_weight * 0.3
                impact_assessment["latency_impact_estimate"] += usage_weight * 0.2
            elif "ECC" in asset.current_algorithm or "ECDSA" in asset.current_algorithm:
                # ECC to post-quantum impact
                impact_assessment["cpu_overhead_estimate"] += usage_weight * 0.4
                impact_assessment["network_overhead_estimate"] += usage_weight * 0.3
            elif "AES" in asset.current_algorithm:
                # AES impact is minimal for quantum-safe
                impact_assessment["cpu_overhead_estimate"] += usage_weight * 0.05
            
            # Identify high-impact operations
            if asset.usage_frequency > 10000 and asset.current_algorithm in ["RSA-2048", "ECDSA-P256"]:
                impact_assessment["high_impact_operations"].append({
                    "asset_id": asset.asset_id,
                    "estimated_impact": "high",
                    "mitigation_needed": True
                })
        
        # Generate mitigation strategies
        if impact_assessment["cpu_overhead_estimate"] > 0.2:
            impact_assessment["mitigation_strategies"].append("Consider hardware acceleration for post-quantum operations")
        
        if impact_assessment["network_overhead_estimate"] > 0.2:
            impact_assessment["mitigation_strategies"].append("Implement certificate compression and caching")
        
        if impact_assessment["latency_impact_estimate"] > 0.15:
            impact_assessment["mitigation_strategies"].append("Pre-compute signatures where possible")
        
        return impact_assessment
    
    def _assess_compliance_requirements(self) -> Dict[str, Any]:
        """Assess compliance requirements for migration"""
        
        compliance_requirements = {}
        all_requirements = set()
        
        for asset in self.cryptographic_assets.values():
            all_requirements.update(asset.compliance_requirements)
        
        for requirement in all_requirements:
            affected_assets = [
                asset.asset_id for asset in self.cryptographic_assets.values()
                if requirement in asset.compliance_requirements
            ]
            
            compliance_requirements[requirement] = {
                "affected_assets": len(affected_assets),
                "asset_list": affected_assets,
                "migration_considerations": self._get_compliance_migration_considerations(requirement)
            }
        
        return compliance_requirements
    
    def _get_compliance_migration_considerations(self, requirement: str) -> List[str]:
        """Get migration considerations for specific compliance requirements"""
        
        considerations = {
            "SOX": [
                "Maintain audit trail integrity during migration",
                "Ensure financial data protection standards",
                "Document all cryptographic changes"
            ],
            "PCI-DSS": [
                "Maintain payment card data protection",
                "Update security policies for new algorithms",
                "Ensure key management compliance"
            ],
            "GDPR": [
                "Maintain personal data protection standards",
                "Document data processing changes",
                "Ensure privacy by design principles"
            ],
            "FINRA": [
                "Maintain trading data integrity",
                "Ensure regulatory reporting compliance",
                "Document risk management changes"
            ],
            "SEC": [
                "Maintain securities trading compliance",
                "Document system security changes",
                "Ensure investor protection standards"
            ]
        }
        
        return considerations.get(requirement, ["Review compliance requirements for this standard"])
    
    def _generate_migration_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate migration recommendations based on assessment"""
        
        recommendations = []
        
        # Overall readiness recommendations
        if assessment["overall_readiness_score"] < 0.5:
            recommendations.append("Overall migration readiness is low - consider comprehensive planning phase")
        elif assessment["overall_readiness_score"] < 0.7:
            recommendations.append("Migration readiness is moderate - address high-risk assets first")
        else:
            recommendations.append("Good migration readiness - consider phased rollout strategy")
        
        # Risk-based recommendations
        high_risk_count = assessment["risk_assessment"]["high_risk_assets"]
        if high_risk_count > 0:
            recommendations.append(f"Priority migration needed for {high_risk_count} high-risk assets")
        
        # Dependency recommendations
        circular_deps = assessment["dependency_analysis"]["circular_dependencies"]
        if circular_deps:
            recommendations.append(f"Resolve {len(circular_deps)} circular dependencies before migration")
        
        # Performance recommendations
        performance = assessment["performance_impact"]
        if performance["cpu_overhead_estimate"] > 0.3:
            recommendations.append("Consider CPU capacity planning for post-quantum algorithms")
        
        if performance["network_overhead_estimate"] > 0.3:
            recommendations.append("Plan for increased network bandwidth requirements")
        
        # Compliance recommendations
        compliance_reqs = len(assessment["compliance_considerations"])
        if compliance_reqs > 3:
            recommendations.append("Engage compliance team early due to multiple regulatory requirements")
        
        return recommendations
    
    async def create_migration_plan(self,
                                  plan_name: str,
                                  strategy: MigrationStrategy,
                                  asset_priorities: Optional[List[str]] = None,
                                  target_completion_date: Optional[datetime] = None,
                                  created_by: str = "system") -> str:
        """Create a comprehensive migration plan"""
        
        plan_id = f"plan-{uuid.uuid4().hex[:16]}"
        
        try:
            # Determine assets to migrate
            if asset_priorities:
                assets_to_migrate = asset_priorities
            else:
                # Auto-prioritize based on risk and urgency
                assets_to_migrate = await self._auto_prioritize_assets()
            
            # Calculate dependencies
            dependencies = self._calculate_plan_dependencies(assets_to_migrate)
            
            # Generate phases based on strategy
            phases = await self._generate_migration_phases(strategy, assets_to_migrate, dependencies)
            
            # Calculate estimates
            estimated_duration = sum(
                (phase.get("estimated_duration", timedelta(0)) for phase in phases),
                timedelta(0)
            )
            
            if not target_completion_date:
                target_completion_date = datetime.now(timezone.utc) + estimated_duration + timedelta(days=7)  # Buffer
            
            # Risk assessment
            risk_assessment = await self._assess_plan_risks(assets_to_migrate, strategy)
            
            # Rollback plan
            rollback_plan = await self._create_rollback_plan(assets_to_migrate)
            
            # Create migration plan
            plan = MigrationPlan(
                plan_id=plan_id,
                plan_name=plan_name,
                strategy=strategy,
                target_completion_date=target_completion_date,
                phases=phases,
                assets_to_migrate=assets_to_migrate,
                dependencies=dependencies,
                risk_assessment=risk_assessment,
                rollback_plan=rollback_plan,
                success_criteria=self._define_success_criteria(assets_to_migrate),
                stakeholders=self._identify_stakeholders(assets_to_migrate),
                estimated_cost=self._estimate_migration_cost(assets_to_migrate),
                estimated_duration=estimated_duration,
                created_by=created_by,
                created_at=datetime.now(timezone.utc)
            )
            
            # Store plan
            self.migration_plans[plan_id] = plan
            
            # Log plan creation
            await self.audit_trail.log_audit_event(
                event_type=AuditEventType.CONFIGURATION_CHANGE,
                user_id=created_by,
                session_id="migration_planning",
                source_system="migration_manager",
                event_data={
                    "action": "migration_plan_created",
                    "plan_id": plan_id,
                    "strategy": strategy.value,
                    "assets_count": len(assets_to_migrate)
                },
                compliance_tags=["migration", "planning"],
                risk_score=0.3
            )
            
            self.logger.info(f"Created migration plan {plan_id}: {plan_name}")
            
            return plan_id
            
        except Exception as e:
            self.logger.error(f"Error creating migration plan: {str(e)}")
            raise
    
    async def _auto_prioritize_assets(self) -> List[str]:
        """Automatically prioritize assets based on quantum risk and business impact"""
        
        asset_scores = []
        
        for asset_id, asset in self.cryptographic_assets.items():
            # Calculate priority score
            priority_score = (
                asset.quantum_vulnerability_score * 0.4 +
                asset.business_impact / 5 * 0.3 +
                asset.criticality_level / 5 * 0.2 +
                (5 - asset.migration_complexity) / 5 * 0.1
            )
            
            asset_scores.append((asset_id, priority_score))
        
        # Sort by priority score (highest first)
        asset_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [asset_id for asset_id, _ in asset_scores]
    
    def _calculate_plan_dependencies(self, assets_to_migrate: List[str]) -> Dict[str, List[str]]:
        """Calculate dependencies for migration plan"""
        
        dependencies = {}
        
        for asset_id in assets_to_migrate:
            asset = self.cryptographic_assets[asset_id]
            asset_dependencies = []
            
            # Find dependencies within the migration scope
            for other_id in assets_to_migrate:
                if other_id != asset_id:
                    other_asset = self.cryptographic_assets[other_id]
                    
                    # Check for system dependencies
                    common_systems = set(asset.dependent_systems) & set(other_asset.dependent_systems)
                    if common_systems:
                        # Create dependency if other asset is foundational
                        if (other_asset.asset_type in ["authentication", "key_exchange"] and
                            asset.asset_type not in ["authentication", "key_exchange"]):
                            asset_dependencies.append(other_id)
            
            dependencies[asset_id] = asset_dependencies
        
        return dependencies
    
    async def _generate_migration_phases(self,
                                       strategy: MigrationStrategy,
                                       assets: List[str],
                                       dependencies: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate migration phases based on strategy"""
        
        phases = []
        
        if strategy == MigrationStrategy.BIG_BANG:
            phases.append({
                "phase_name": "complete_migration",
                "description": "Migrate all assets simultaneously",
                "assets": assets,
                "estimated_duration": timedelta(hours=24),
                "parallel_execution": True,
                "rollback_plan": "full_system_rollback",
                "validation_steps": ["complete_system_validation"]
            })
        
        elif strategy == MigrationStrategy.PHASED:
            # Create phases based on dependency waves
            dependency_graph = {aid: dependencies.get(aid, []) for aid in assets}
            waves = self._calculate_migration_waves(dependency_graph)
            
            for i, wave in enumerate(waves):
                phases.append({
                    "phase_name": f"phase_{i+1}",
                    "description": f"Migrate wave {i+1} assets",
                    "assets": wave,
                    "estimated_duration": timedelta(hours=8 * len(wave)),
                    "parallel_execution": len(wave) > 1,
                    "dependencies": [f"phase_{i}" for i in range(i)],
                    "rollback_plan": f"rollback_phase_{i+1}",
                    "validation_steps": [f"validate_phase_{i+1}"]
                })
        
        elif strategy == MigrationStrategy.CANARY:
            # Start with lowest risk assets
            sorted_assets = sorted(assets, key=lambda aid: self.cryptographic_assets[aid].business_impact)
            
            canary_size = max(1, len(assets) // 10)  # 10% canary
            
            phases.append({
                "phase_name": "canary_deployment",
                "description": "Deploy to canary assets",
                "assets": sorted_assets[:canary_size],
                "estimated_duration": timedelta(hours=4),
                "parallel_execution": False,
                "rollback_plan": "canary_rollback",
                "validation_steps": ["canary_validation", "performance_monitoring"]
            })
            
            phases.append({
                "phase_name": "full_rollout",
                "description": "Full rollout after canary validation",
                "assets": sorted_assets[canary_size:],
                "estimated_duration": timedelta(hours=12),
                "parallel_execution": True,
                "dependencies": ["canary_deployment"],
                "rollback_plan": "full_rollback",
                "validation_steps": ["complete_validation"]
            })
        
        # Add additional phases based on templates
        for phase in phases:
            phase["validation_steps"] = phase.get("validation_steps", []) + [
                "security_validation",
                "performance_verification",
                "compliance_check"
            ]
        
        return phases
    
    async def _assess_plan_risks(self, assets: List[str], strategy: MigrationStrategy) -> Dict[str, Any]:
        """Assess risks for migration plan"""
        
        risk_assessment = {
            "overall_risk_level": "medium",
            "risk_factors": [],
            "mitigation_strategies": [],
            "contingency_plans": []
        }
        
        # Assess asset-specific risks
        high_risk_assets = [
            aid for aid in assets 
            if self.cryptographic_assets[aid].business_impact >= 4
        ]
        
        if len(high_risk_assets) > len(assets) * 0.5:
            risk_assessment["risk_factors"].append("High proportion of business-critical assets")
            risk_assessment["overall_risk_level"] = "high"
        
        # Strategy-specific risks
        if strategy == MigrationStrategy.BIG_BANG:
            risk_assessment["risk_factors"].append("Big bang migration increases failure impact")
            risk_assessment["mitigation_strategies"].append("Extensive pre-migration testing required")
        
        elif strategy == MigrationStrategy.PHASED:
            risk_assessment["risk_factors"].append("Extended migration period increases exposure window")
            risk_assessment["mitigation_strategies"].append("Hybrid security controls during transition")
        
        # Performance risks
        performance_sensitive_assets = [
            aid for aid in assets 
            if self.cryptographic_assets[aid].usage_frequency > 10000
        ]
        
        if performance_sensitive_assets:
            risk_assessment["risk_factors"].append("Performance-sensitive assets may impact user experience")
            risk_assessment["mitigation_strategies"].append("Performance monitoring and capacity planning")
        
        return risk_assessment
    
    async def _create_rollback_plan(self, assets: List[str]) -> Dict[str, Any]:
        """Create rollback plan for migration"""
        
        rollback_plan = {
            "rollback_triggers": [
                "performance_degradation_exceeds_threshold",
                "security_validation_failure",
                "business_critical_failure",
                "compliance_violation_detected"
            ],
            "rollback_procedures": {},
            "rollback_time_estimate": timedelta(hours=4),
            "data_preservation": True,
            "validation_after_rollback": [
                "system_functionality_check",
                "security_verification",
                "performance_baseline_restoration"
            ]
        }
        
        # Asset-specific rollback procedures
        for asset_id in assets:
            asset = self.cryptographic_assets[asset_id]
            rollback_plan["rollback_procedures"][asset_id] = {
                "restore_algorithm": asset.current_algorithm,
                "restore_keys": True,
                "update_configurations": True,
                "restart_services": asset.dependent_systems,
                "estimated_time": timedelta(minutes=30)
            }
        
        return rollback_plan
    
    def _define_success_criteria(self, assets: List[str]) -> List[str]:
        """Define success criteria for migration"""
        
        criteria = [
            "All migrated assets use post-quantum algorithms",
            "System functionality maintained or improved",
            "Performance degradation within acceptable limits",
            "Security validation passes all tests",
            "Compliance requirements satisfied",
            "No critical incidents during migration window"
        ]
        
        # Asset-specific criteria
        for asset_id in assets:
            asset = self.cryptographic_assets[asset_id]
            if asset.business_impact >= 4:
                criteria.append(f"Zero downtime for {asset.asset_name}")
        
        return criteria
    
    def _identify_stakeholders(self, assets: List[str]) -> List[str]:
        """Identify stakeholders for migration"""
        
        stakeholders = ["security_team", "operations_team", "development_team"]
        
        # Add stakeholders based on asset types
        asset_types = set(self.cryptographic_assets[aid].asset_type for aid in assets)
        
        if "authentication" in asset_types:
            stakeholders.append("identity_management_team")
        
        compliance_reqs = set()
        for asset_id in assets:
            compliance_reqs.update(self.cryptographic_assets[asset_id].compliance_requirements)
        
        if compliance_reqs:
            stakeholders.append("compliance_team")
        
        if any(self.cryptographic_assets[aid].business_impact >= 4 for aid in assets):
            stakeholders.extend(["business_stakeholders", "risk_management"])
        
        return stakeholders
    
    def _estimate_migration_cost(self, assets: List[str]) -> float:
        """Estimate migration cost"""
        
        base_cost_per_hour = 200.0  # Average team cost per hour
        total_cost = 0.0
        
        for asset_id in assets:
            asset = self.cryptographic_assets[asset_id]
            
            # Cost factors
            complexity_multiplier = asset.migration_complexity / 5
            dependency_multiplier = len(asset.dependent_systems) / 10
            testing_multiplier = asset.business_impact / 5
            
            estimated_hours = (
                asset.estimated_migration_time.total_seconds() / 3600 * 
                (1 + complexity_multiplier + dependency_multiplier + testing_multiplier)
            )
            
            asset_cost = estimated_hours * base_cost_per_hour
            total_cost += asset_cost
        
        return total_cost
    
    async def _migration_monitoring_loop(self):
        """Background monitoring loop for active migrations"""
        
        while not self._shutdown_event.is_set():
            try:
                # Monitor active executions
                for execution_id, execution in self.active_executions.items():
                    if execution.status == MigrationStatus.IN_PROGRESS:
                        # Update progress and check for issues
                        await self._update_execution_progress(execution)
                        
                        # Check for auto-rollback conditions
                        if self.enable_auto_rollback:
                            await self._check_rollback_conditions(execution)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in migration monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def _update_execution_progress(self, execution: MigrationExecution):
        """Update progress for active migration execution"""
        
        # Simplified progress tracking
        plan = self.migration_plans[execution.plan_id]
        total_phases = len(plan.phases)
        completed_phases = len(execution.completed_phases)
        
        if total_phases > 0:
            execution.progress_percentage = (completed_phases / total_phases) * 100
        
        # Update estimated completion
        if execution.progress_percentage > 0:
            elapsed_time = datetime.now(timezone.utc) - execution.start_time
            estimated_total_time = elapsed_time / (execution.progress_percentage / 100)
            execution.estimated_completion = execution.start_time + estimated_total_time
    
    async def _check_rollback_conditions(self, execution: MigrationExecution):
        """Check if rollback conditions are met"""
        
        rollback_needed = False
        rollback_reason = ""
        
        # Check performance degradation
        if execution.performance_metrics.get("degradation", 0.0) > self.performance_threshold:
            rollback_needed = True
            rollback_reason = f"Performance degradation exceeds threshold: {execution.performance_metrics['degradation']:.2%}"
        
        # Check for failed steps
        if len(execution.failed_steps) > 3:  # Threshold for maximum failures
            rollback_needed = True
            rollback_reason = f"Too many failed steps: {len(execution.failed_steps)}"
        
        if rollback_needed:
            self.logger.warning(f"Auto-rollback triggered for execution {execution.execution_id}: {rollback_reason}")
            await self._initiate_rollback(execution.execution_id, rollback_reason)
    
    async def _initiate_rollback(self, execution_id: str, reason: str):
        """Initiate rollback for a migration execution"""
        
        if execution_id not in self.active_executions:
            return
        
        execution = self.active_executions[execution_id]
        execution.status = MigrationStatus.ROLLBACK
        
        try:
            # Execute rollback plan
            plan = self.migration_plans[execution.plan_id]
            rollback_plan = plan.rollback_plan
            
            # Log rollback initiation
            await self.audit_trail.log_audit_event(
                event_type=AuditEventType.SYSTEM_ACCESS,
                user_id="system",
                session_id="migration_rollback",
                source_system="migration_manager",
                event_data={
                    "action": "migration_rollback_initiated",
                    "execution_id": execution_id,
                    "reason": reason,
                    "rollback_plan": rollback_plan
                },
                compliance_tags=["migration", "rollback", "security"],
                risk_score=0.7
            )
            
            # Execute rollback procedures
            for asset_id in plan.assets_to_migrate:
                await self._rollback_asset(asset_id, rollback_plan["rollback_procedures"][asset_id])
            
            execution.status = MigrationStatus.ROLLED_BACK
            self.migration_metrics["rollbacks_performed"] += 1
            
            self.logger.info(f"Migration rollback completed for execution {execution_id}")
            
        except Exception as e:
            self.logger.error(f"Error during rollback for execution {execution_id}: {str(e)}")
            execution.status = MigrationStatus.FAILED
    
    async def _rollback_asset(self, asset_id: str, rollback_procedure: Dict[str, Any]):
        """Rollback a specific asset to its previous state"""
        
        # Simplified rollback implementation
        # In practice, would restore configurations, keys, and restart services
        
        self.logger.info(f"Rolling back asset {asset_id}")
        
        # Simulate rollback time
        await asyncio.sleep(rollback_procedure.get("estimated_time", timedelta(minutes=5)).total_seconds())
        
        self.logger.info(f"Asset {asset_id} rollback completed")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        
        while not self._shutdown_event.is_set():
            try:
                # Update migration metrics
                total_assets = len(self.cryptographic_assets)
                migrated_assets = sum(
                    1 for asset in self.cryptographic_assets.values()
                    if asset.metadata.get("migrated", False)
                )
                
                self.migration_metrics.update({
                    "total_assets": total_assets,
                    "migrated_assets": migrated_assets,
                    "migration_completion_rate": migrated_assets / max(total_assets, 1)
                })
                
                # Calculate success rate
                total_executions = len(self.migration_plans)
                successful_executions = sum(
                    1 for exec in self.active_executions.values()
                    if exec.status == MigrationStatus.COMPLETED
                )
                
                if total_executions > 0:
                    self.migration_metrics["success_rate"] = successful_executions / total_executions
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {str(e)}")
                await asyncio.sleep(300)
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration system status"""
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_status": "operational",
            "assets": {
                "total_assets": len(self.cryptographic_assets),
                "high_priority_assets": len([
                    a for a in self.cryptographic_assets.values()
                    if a.quantum_vulnerability_score > 0.8
                ]),
                "migration_ready": len([
                    a for a in self.cryptographic_assets.values()
                    if a.migration_complexity <= 3
                ])
            },
            "migration_plans": {
                "total_plans": len(self.migration_plans),
                "active_executions": len([
                    e for e in self.active_executions.values()
                    if e.status == MigrationStatus.IN_PROGRESS
                ])
            },
            "performance_metrics": self.migration_metrics,
            "templates_available": len(self.migration_templates),
            "patterns_loaded": len(self.migration_patterns)
        }
    
    async def shutdown(self):
        """Shutdown migration manager gracefully"""
        
        self.logger.info("Shutting down quantum-safe migration manager")
        
        self._shutdown_event.set()
        self.executor.shutdown(wait=True)
        
        # Pause any active migrations
        for execution in self.active_executions.values():
            if execution.status == MigrationStatus.IN_PROGRESS:
                execution.status = MigrationStatus.PAUSED
        
        self.logger.info("Migration manager shutdown complete")