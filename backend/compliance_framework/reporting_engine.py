"""
Automated Compliance Reporting Engine
====================================

Advanced reporting system for regulatory compliance including SOC 2,
Basel III, GDPR, and other international compliance frameworks.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import uuid
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import yaml
import schedule


class ReportType(Enum):
    """Compliance report types"""
    SOC_2_TYPE_II = "soc_2_type_ii"
    BASEL_III_CAPITAL = "basel_iii_capital"
    GDPR_PRIVACY_IMPACT = "gdpr_privacy_impact"
    FINRA_SURVEILLANCE = "finra_surveillance"
    AML_SUSPICIOUS_ACTIVITY = "aml_suspicious_activity"
    CYBERSECURITY_FRAMEWORK = "cybersecurity_framework"
    BUSINESS_CONTINUITY = "business_continuity"
    VENDOR_RISK_ASSESSMENT = "vendor_risk_assessment"
    DATA_GOVERNANCE = "data_governance"
    OPERATIONAL_RISK = "operational_risk"


class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    EXCEL = "xlsx"
    CSV = "csv"
    XML = "xml"


class ReportFrequency(Enum):
    """Report generation frequency"""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


@dataclass
class ReportConfiguration:
    """Configuration for compliance reports"""
    report_id: str
    name: str
    report_type: ReportType
    description: str
    format: ReportFormat
    frequency: ReportFrequency
    recipients: List[str]
    jurisdiction: str
    regulatory_body: str
    template_path: str
    data_sources: List[str]
    filters: Dict[str, Any]
    custom_parameters: Dict[str, Any]
    retention_days: int
    encryption_required: bool
    digital_signature_required: bool
    compliance_framework: str
    risk_rating: str
    business_owner: str
    technical_owner: str
    created_at: datetime
    updated_at: datetime
    next_generation: datetime
    is_active: bool


@dataclass
class ReportMetadata:
    """Metadata for generated reports"""
    report_id: str
    configuration_id: str
    generation_id: str
    report_type: ReportType
    format: ReportFormat
    generated_at: datetime
    generation_time_seconds: float
    data_period_start: datetime
    data_period_end: datetime
    jurisdiction: str
    regulatory_body: str
    total_records: int
    filtered_records: int
    data_sources_used: List[str]
    compliance_status: str
    risk_level: str
    findings_count: int
    recommendations_count: int
    file_path: str
    file_size_bytes: int
    checksum: str
    digital_signature: str
    encryption_status: str
    recipients: List[str]
    distribution_status: str


class AutomatedComplianceReporter:
    """
    Automated compliance reporting engine that generates regulatory
    reports according to various international compliance frameworks.
    """
    
    def __init__(self, 
                 data_directory: str = "/app/compliance/reports",
                 template_directory: str = "/app/compliance/templates"):
        self.data_directory = Path(data_directory)
        self.template_directory = Path(template_directory)
        
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.template_directory.mkdir(parents=True, exist_ok=True)
        
        self.configurations: Dict[str, ReportConfiguration] = {}
        self.report_metadata: Dict[str, ReportMetadata] = {}
        self.generation_queue: List[str] = []
        
        self.logger = logging.getLogger("compliance.reporting")
        
        # Initialize Jinja2 environment for templating
        self.jinja_env = Environment(loader=FileSystemLoader(str(self.template_directory)))
        
        # Initialize default configurations
        self._initialize_default_configurations()
        
        # Start scheduler
        self._start_scheduler()
    
    def _initialize_default_configurations(self):
        """Initialize default report configurations"""
        
        # SOC 2 Type II Report
        soc2_config = ReportConfiguration(
            report_id="SOC2-001",
            name="SOC 2 Type II Controls Assessment",
            report_type=ReportType.SOC_2_TYPE_II,
            description="Comprehensive SOC 2 Type II assessment of security, availability, processing integrity, confidentiality, and privacy controls",
            format=ReportFormat.PDF,
            frequency=ReportFrequency.QUARTERLY,
            recipients=["compliance@nautilus.com", "audit@nautilus.com", "cso@nautilus.com"],
            jurisdiction="US",
            regulatory_body="AICPA",
            template_path="soc2_type_ii_template.html",
            data_sources=["audit_trail", "security_events", "access_controls", "system_monitoring"],
            filters={
                "date_range_months": 12,
                "control_categories": ["security", "availability", "processing_integrity", "confidentiality", "privacy"],
                "severity_levels": ["medium", "high", "critical"]
            },
            custom_parameters={
                "auditor_firm": "External Audit Firm",
                "examination_period": "12 months",
                "control_testing_frequency": "quarterly"
            },
            retention_days=2555,  # 7 years
            encryption_required=True,
            digital_signature_required=True,
            compliance_framework="SOC 2",
            risk_rating="high",
            business_owner="Chief Compliance Officer",
            technical_owner="Information Security Manager",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            next_generation=datetime.now(timezone.utc) + timedelta(days=90),
            is_active=True
        )
        self.add_configuration(soc2_config)
        
        # Basel III Capital Report
        basel_config = ReportConfiguration(
            report_id="BASEL-001",
            name="Basel III Capital Adequacy Assessment",
            report_type=ReportType.BASEL_III_CAPITAL,
            description="Basel III capital ratios, leverage ratios, and liquidity coverage analysis",
            format=ReportFormat.EXCEL,
            frequency=ReportFrequency.MONTHLY,
            recipients=["riskmanager@nautilus.com", "cfo@nautilus.com", "regulator@centralbank.gov"],
            jurisdiction="GLOBAL",
            regulatory_body="Basel Committee on Banking Supervision",
            template_path="basel_iii_template.xlsx",
            data_sources=["trading_positions", "capital_calculations", "risk_metrics", "liquidity_ratios"],
            filters={
                "calculation_date": "month_end",
                "capital_types": ["cet1", "tier1", "total_capital"],
                "risk_categories": ["credit", "market", "operational"]
            },
            custom_parameters={
                "cet1_minimum": 4.5,
                "tier1_minimum": 6.0,
                "total_capital_minimum": 8.0,
                "leverage_ratio_minimum": 3.0,
                "lcr_minimum": 100.0
            },
            retention_days=3650,  # 10 years
            encryption_required=True,
            digital_signature_required=True,
            compliance_framework="Basel III",
            risk_rating="critical",
            business_owner="Chief Risk Officer",
            technical_owner="Risk Analytics Manager",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            next_generation=datetime.now(timezone.utc) + timedelta(days=30),
            is_active=True
        )
        self.add_configuration(basel_config)
        
        # GDPR Privacy Impact Assessment
        gdpr_config = ReportConfiguration(
            report_id="GDPR-001",
            name="GDPR Data Protection Impact Assessment",
            report_type=ReportType.GDPR_PRIVACY_IMPACT,
            description="Comprehensive GDPR compliance assessment including data processing activities, privacy controls, and breach analysis",
            format=ReportFormat.HTML,
            frequency=ReportFrequency.QUARTERLY,
            recipients=["dpo@nautilus.com", "privacy@nautilus.com", "legal@nautilus.com"],
            jurisdiction="EU",
            regulatory_body="European Data Protection Board",
            template_path="gdpr_dpia_template.html",
            data_sources=["personal_data_inventory", "consent_records", "data_breaches", "subject_requests"],
            filters={
                "data_categories": ["personal", "special_category"],
                "processing_purposes": ["trading", "marketing", "analytics"],
                "legal_bases": ["consent", "contract", "legitimate_interest"]
            },
            custom_parameters={
                "dpo_contact": "dpo@nautilus.com",
                "supervisory_authority": "Local Data Protection Authority",
                "privacy_policy_version": "2.1"
            },
            retention_days=2190,  # 6 years
            encryption_required=True,
            digital_signature_required=True,
            compliance_framework="GDPR",
            risk_rating="high",
            business_owner="Data Protection Officer",
            technical_owner="Privacy Engineer",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            next_generation=datetime.now(timezone.utc) + timedelta(days=90),
            is_active=True
        )
        self.add_configuration(gdpr_config)
        
        # FINRA Market Surveillance Report
        finra_config = ReportConfiguration(
            report_id="FINRA-001",
            name="FINRA Market Surveillance and Trade Monitoring",
            report_type=ReportType.FINRA_SURVEILLANCE,
            description="Real-time market surveillance, trade reporting, and compliance monitoring for FINRA requirements",
            format=ReportFormat.JSON,
            frequency=ReportFrequency.DAILY,
            recipients=["compliance@nautilus.com", "surveillance@finra.org"],
            jurisdiction="US",
            regulatory_body="FINRA",
            template_path="finra_surveillance_template.json",
            data_sources=["trade_executions", "order_book", "market_surveillance", "exception_reports"],
            filters={
                "trade_types": ["equity", "option", "fixed_income"],
                "surveillance_patterns": ["layering", "spoofing", "wash_trading", "insider_trading"],
                "exception_categories": ["best_execution", "order_handling", "market_making"]
            },
            custom_parameters={
                "firm_crd": "123456",
                "surveillance_system": "Nautilus Advanced Surveillance",
                "reporting_format": "FINRA CAT"
            },
            retention_days=2190,  # 6 years
            encryption_required=True,
            digital_signature_required=False,
            compliance_framework="FINRA Rules",
            risk_rating="critical",
            business_owner="Chief Compliance Officer",
            technical_owner="Market Surveillance Manager",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            next_generation=datetime.now(timezone.utc) + timedelta(days=1),
            is_active=True
        )
        self.add_configuration(finra_config)
    
    def add_configuration(self, config: ReportConfiguration) -> bool:
        """Add a report configuration"""
        try:
            self.configurations[config.report_id] = config
            self._save_configuration(config)
            self.logger.info(f"Added report configuration: {config.name} ({config.report_id})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add configuration {config.report_id}: {str(e)}")
            return False
    
    def update_configuration(self, report_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing report configuration"""
        if report_id not in self.configurations:
            self.logger.error(f"Configuration {report_id} not found")
            return False
        
        try:
            config = self.configurations[report_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(config, field):
                    setattr(config, field, value)
            
            config.updated_at = datetime.now(timezone.utc)
            
            self._save_configuration(config)
            self.logger.info(f"Updated report configuration: {report_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration {report_id}: {str(e)}")
            return False
    
    async def generate_report(self, report_id: str, 
                            custom_period: Optional[Tuple[datetime, datetime]] = None) -> Optional[ReportMetadata]:
        """Generate a compliance report"""
        
        if report_id not in self.configurations:
            self.logger.error(f"Report configuration {report_id} not found")
            return None
        
        config = self.configurations[report_id]
        generation_start = datetime.now(timezone.utc)
        generation_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Starting report generation: {config.name} ({generation_id})")
            
            # Determine data period
            if custom_period:
                period_start, period_end = custom_period
            else:
                period_start, period_end = self._calculate_data_period(config)
            
            # Collect data
            report_data = await self._collect_report_data(config, period_start, period_end)
            
            # Process data according to compliance framework
            processed_data = await self._process_compliance_data(config, report_data)
            
            # Generate report content
            report_content = await self._generate_report_content(config, processed_data)
            
            # Create report file
            file_path = await self._create_report_file(config, report_content, generation_id)
            
            # Create metadata
            generation_end = datetime.now(timezone.utc)
            generation_time = (generation_end - generation_start).total_seconds()
            
            metadata = ReportMetadata(
                report_id=report_id,
                configuration_id=config.report_id,
                generation_id=generation_id,
                report_type=config.report_type,
                format=config.format,
                generated_at=generation_end,
                generation_time_seconds=generation_time,
                data_period_start=period_start,
                data_period_end=period_end,
                jurisdiction=config.jurisdiction,
                regulatory_body=config.regulatory_body,
                total_records=report_data.get("total_records", 0),
                filtered_records=report_data.get("filtered_records", 0),
                data_sources_used=config.data_sources,
                compliance_status=processed_data.get("compliance_status", "unknown"),
                risk_level=processed_data.get("risk_level", "unknown"),
                findings_count=len(processed_data.get("findings", [])),
                recommendations_count=len(processed_data.get("recommendations", [])),
                file_path=str(file_path),
                file_size_bytes=file_path.stat().st_size,
                checksum=self._calculate_file_checksum(file_path),
                digital_signature=await self._create_digital_signature(file_path) if config.digital_signature_required else "",
                encryption_status="encrypted" if config.encryption_required else "unencrypted",
                recipients=config.recipients,
                distribution_status="pending"
            )
            
            self.report_metadata[generation_id] = metadata
            await self._save_report_metadata(metadata)
            
            # Distribute report
            if config.recipients:
                await self._distribute_report(metadata)
            
            # Update next generation time
            config.next_generation = self._calculate_next_generation(config)
            await self._save_configuration(config)
            
            self.logger.info(f"Report generation completed: {generation_id} in {generation_time:.2f}s")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Report generation failed for {report_id}: {str(e)}")
            return None
    
    async def _collect_report_data(self, config: ReportConfiguration, 
                                 period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Collect data from various sources for report generation"""
        
        collected_data = {
            "total_records": 0,
            "filtered_records": 0,
            "data_sources": {}
        }
        
        for source in config.data_sources:
            try:
                source_data = await self._collect_from_source(source, config.filters, period_start, period_end)
                collected_data["data_sources"][source] = source_data
                collected_data["total_records"] += source_data.get("record_count", 0)
            except Exception as e:
                self.logger.error(f"Failed to collect data from {source}: {str(e)}")
                collected_data["data_sources"][source] = {"error": str(e), "record_count": 0}
        
        collected_data["filtered_records"] = collected_data["total_records"]
        return collected_data
    
    async def _collect_from_source(self, source: str, filters: Dict[str, Any], 
                                 start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect data from a specific source"""
        
        # This would interface with actual data sources
        # For now, return mock data structure
        
        if source == "audit_trail":
            return {
                "record_count": 15420,
                "events": [
                    {
                        "timestamp": "2024-08-23T10:30:00Z",
                        "event_type": "user_login",
                        "user_id": "user123",
                        "risk_level": "low"
                    }
                    # ... more events
                ],
                "compliance_events": 234,
                "security_events": 45,
                "violations": 3
            }
        
        elif source == "trading_positions":
            return {
                "record_count": 8456,
                "positions": [
                    {
                        "symbol": "AAPL",
                        "quantity": 1000,
                        "market_value": 150000,
                        "risk_weight": 0.25
                    }
                    # ... more positions
                ],
                "total_exposure": 25000000,
                "risk_weighted_assets": 15000000
            }
        
        elif source == "personal_data_inventory":
            return {
                "record_count": 2341,
                "data_subjects": 12450,
                "processing_activities": [
                    {
                        "purpose": "trading_services",
                        "legal_basis": "contract",
                        "data_categories": ["contact", "financial"],
                        "retention_period": "7_years"
                    }
                    # ... more activities
                ],
                "consent_records": 11890,
                "data_breaches": 1
            }
        
        else:
            return {
                "record_count": 0,
                "error": f"Unknown data source: {source}"
            }
    
    async def _process_compliance_data(self, config: ReportConfiguration, 
                                     raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw data according to compliance framework requirements"""
        
        processed = {
            "compliance_status": "compliant",
            "risk_level": "low",
            "findings": [],
            "recommendations": [],
            "metrics": {},
            "framework_specific": {}
        }
        
        if config.report_type == ReportType.SOC_2_TYPE_II:
            processed = await self._process_soc2_data(raw_data)
        elif config.report_type == ReportType.BASEL_III_CAPITAL:
            processed = await self._process_basel_data(raw_data)
        elif config.report_type == ReportType.GDPR_PRIVACY_IMPACT:
            processed = await self._process_gdpr_data(raw_data)
        elif config.report_type == ReportType.FINRA_SURVEILLANCE:
            processed = await self._process_finra_data(raw_data)
        
        return processed
    
    async def _process_soc2_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for SOC 2 Type II report"""
        
        audit_data = raw_data["data_sources"].get("audit_trail", {})
        security_data = raw_data["data_sources"].get("security_events", {})
        
        findings = []
        if audit_data.get("violations", 0) > 0:
            findings.append({
                "control_id": "CC6.1",
                "description": "Logical access violations detected",
                "severity": "medium",
                "count": audit_data.get("violations", 0)
            })
        
        recommendations = []
        if security_data.get("failed_logins", 0) > 100:
            recommendations.append({
                "area": "Access Controls",
                "recommendation": "Implement additional authentication controls",
                "priority": "high"
            })
        
        return {
            "compliance_status": "compliant" if len(findings) == 0 else "non_compliant",
            "risk_level": "high" if any(f["severity"] == "high" for f in findings) else "medium",
            "findings": findings,
            "recommendations": recommendations,
            "metrics": {
                "total_controls_tested": 75,
                "controls_passed": 73,
                "controls_failed": 2,
                "testing_coverage": "97.3%"
            },
            "framework_specific": {
                "trust_services_criteria": {
                    "security": {"status": "effective", "exceptions": 1},
                    "availability": {"status": "effective", "exceptions": 0},
                    "processing_integrity": {"status": "effective", "exceptions": 1},
                    "confidentiality": {"status": "effective", "exceptions": 0},
                    "privacy": {"status": "effective", "exceptions": 0}
                }
            }
        }
    
    async def _process_basel_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for Basel III capital report"""
        
        position_data = raw_data["data_sources"].get("trading_positions", {})
        
        total_exposure = position_data.get("total_exposure", 0)
        risk_weighted_assets = position_data.get("risk_weighted_assets", 0)
        
        # Mock capital calculations
        cet1_capital = 2000000  # $2M
        tier1_capital = 2500000  # $2.5M
        total_capital = 3000000  # $3M
        
        cet1_ratio = (cet1_capital / risk_weighted_assets) * 100 if risk_weighted_assets > 0 else 0
        tier1_ratio = (tier1_capital / risk_weighted_assets) * 100 if risk_weighted_assets > 0 else 0
        total_capital_ratio = (total_capital / risk_weighted_assets) * 100 if risk_weighted_assets > 0 else 0
        
        findings = []
        if cet1_ratio < 4.5:
            findings.append({
                "requirement": "CET1 Ratio",
                "actual": f"{cet1_ratio:.2f}%",
                "required": "4.5%",
                "status": "non_compliant"
            })
        
        return {
            "compliance_status": "compliant" if len(findings) == 0 else "non_compliant",
            "risk_level": "critical" if any(f["status"] == "non_compliant" for f in findings) else "low",
            "findings": findings,
            "recommendations": [],
            "metrics": {
                "cet1_ratio": f"{cet1_ratio:.2f}%",
                "tier1_ratio": f"{tier1_ratio:.2f}%",
                "total_capital_ratio": f"{total_capital_ratio:.2f}%",
                "leverage_ratio": "5.2%",
                "lcr": "110.5%"
            },
            "framework_specific": {
                "capital_buffers": {
                    "capital_conservation_buffer": "2.5%",
                    "countercyclical_buffer": "0.0%",
                    "systemic_buffer": "0.0%"
                },
                "liquidity_metrics": {
                    "lcr": {"value": "110.5%", "minimum": "100%", "status": "compliant"},
                    "nsfr": {"value": "108.2%", "minimum": "100%", "status": "compliant"}
                }
            }
        }
    
    async def _process_gdpr_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for GDPR privacy impact assessment"""
        
        privacy_data = raw_data["data_sources"].get("personal_data_inventory", {})
        
        findings = []
        if privacy_data.get("data_breaches", 0) > 0:
            findings.append({
                "requirement": "Data Breach Management",
                "description": "Data breach incidents recorded",
                "count": privacy_data.get("data_breaches", 0),
                "severity": "high"
            })
        
        return {
            "compliance_status": "compliant" if len(findings) == 0 else "requires_attention",
            "risk_level": "medium",
            "findings": findings,
            "recommendations": [
                "Conduct privacy training for all staff",
                "Review data retention policies",
                "Update privacy notices"
            ],
            "metrics": {
                "data_subjects": privacy_data.get("data_subjects", 0),
                "processing_activities": len(privacy_data.get("processing_activities", [])),
                "consent_rate": "95.6%",
                "data_subject_requests": 45
            },
            "framework_specific": {
                "lawful_bases": {
                    "consent": "67%",
                    "contract": "25%",
                    "legitimate_interest": "8%"
                },
                "data_categories": {
                    "personal_identifiers": "100%",
                    "financial_data": "78%",
                    "contact_information": "100%",
                    "special_categories": "0%"
                }
            }
        }
    
    async def _process_finra_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for FINRA surveillance report"""
        
        # This would process actual trading surveillance data
        return {
            "compliance_status": "compliant",
            "risk_level": "low",
            "findings": [],
            "recommendations": [],
            "metrics": {
                "trades_monitored": 145620,
                "alerts_generated": 234,
                "investigations_opened": 12,
                "violations_found": 0
            },
            "framework_specific": {
                "surveillance_patterns": {
                    "layering": {"alerts": 45, "investigations": 3, "violations": 0},
                    "spoofing": {"alerts": 23, "investigations": 2, "violations": 0},
                    "wash_trading": {"alerts": 12, "investigations": 1, "violations": 0},
                    "insider_trading": {"alerts": 156, "investigations": 6, "violations": 0}
                }
            }
        }
    
    async def _generate_report_content(self, config: ReportConfiguration, 
                                     processed_data: Dict[str, Any]) -> str:
        """Generate report content using templates"""
        
        try:
            template = self.jinja_env.get_template(config.template_path)
            
            context = {
                "config": config,
                "data": processed_data,
                "generation_date": datetime.now(timezone.utc),
                "report_period": {
                    "start": processed_data.get("period_start", "Unknown"),
                    "end": processed_data.get("period_end", "Unknown")
                }
            }
            
            content = template.render(**context)
            return content
            
        except Exception as e:
            self.logger.error(f"Template rendering failed: {str(e)}")
            # Return JSON fallback
            return json.dumps(processed_data, indent=2, default=str)
    
    async def _create_report_file(self, config: ReportConfiguration, 
                                content: str, generation_id: str) -> Path:
        """Create report file in specified format"""
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{config.report_id}_{timestamp}_{generation_id[:8]}"
        
        if config.format == ReportFormat.JSON:
            file_path = self.data_directory / f"{filename}.json"
            with open(file_path, 'w') as f:
                f.write(content)
        
        elif config.format == ReportFormat.HTML:
            file_path = self.data_directory / f"{filename}.html"
            with open(file_path, 'w') as f:
                f.write(content)
        
        elif config.format == ReportFormat.PDF:
            # In production, would use proper PDF generation
            file_path = self.data_directory / f"{filename}.pdf"
            with open(file_path, 'w') as f:
                f.write(f"PDF Content: {content}")
        
        elif config.format == ReportFormat.EXCEL:
            # In production, would use proper Excel generation
            file_path = self.data_directory / f"{filename}.xlsx"
            with open(file_path, 'w') as f:
                f.write(f"Excel Content: {content}")
        
        else:
            file_path = self.data_directory / f"{filename}.txt"
            with open(file_path, 'w') as f:
                f.write(content)
        
        return file_path
    
    def _calculate_data_period(self, config: ReportConfiguration) -> Tuple[datetime, datetime]:
        """Calculate data period for report"""
        
        now = datetime.now(timezone.utc)
        
        if config.frequency == ReportFrequency.DAILY:
            start = now - timedelta(days=1)
            end = now
        elif config.frequency == ReportFrequency.WEEKLY:
            start = now - timedelta(weeks=1)
            end = now
        elif config.frequency == ReportFrequency.MONTHLY:
            start = now - timedelta(days=30)
            end = now
        elif config.frequency == ReportFrequency.QUARTERLY:
            start = now - timedelta(days=90)
            end = now
        elif config.frequency == ReportFrequency.ANNUALLY:
            start = now - timedelta(days=365)
            end = now
        else:
            start = now - timedelta(days=1)
            end = now
        
        return start, end
    
    def _calculate_next_generation(self, config: ReportConfiguration) -> datetime:
        """Calculate next report generation time"""
        
        now = datetime.now(timezone.utc)
        
        if config.frequency == ReportFrequency.REAL_TIME:
            return now + timedelta(minutes=5)
        elif config.frequency == ReportFrequency.HOURLY:
            return now + timedelta(hours=1)
        elif config.frequency == ReportFrequency.DAILY:
            return now + timedelta(days=1)
        elif config.frequency == ReportFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif config.frequency == ReportFrequency.MONTHLY:
            return now + timedelta(days=30)
        elif config.frequency == ReportFrequency.QUARTERLY:
            return now + timedelta(days=90)
        elif config.frequency == ReportFrequency.ANNUALLY:
            return now + timedelta(days=365)
        else:
            return now + timedelta(days=1)
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum for integrity verification"""
        import hashlib
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _create_digital_signature(self, file_path: Path) -> str:
        """Create digital signature for report file"""
        # In production, would use proper digital signing
        return f"SIGNATURE_{file_path.name}_{datetime.now(timezone.utc).timestamp()}"
    
    async def _distribute_report(self, metadata: ReportMetadata):
        """Distribute report to configured recipients"""
        
        try:
            # In production, would implement actual distribution (email, API, etc.)
            self.logger.info(f"Distributing report {metadata.generation_id} to {len(metadata.recipients)} recipients")
            
            # Mock distribution success
            metadata.distribution_status = "completed"
            
        except Exception as e:
            self.logger.error(f"Report distribution failed: {str(e)}")
            metadata.distribution_status = "failed"
    
    def _save_configuration(self, config: ReportConfiguration):
        """Save report configuration to storage"""
        config_file = self.data_directory / f"config_{config.report_id}.json"
        
        config_dict = asdict(config)
        
        # Convert datetime objects
        for key, value in config_dict.items():
            if isinstance(value, datetime):
                config_dict[key] = value.isoformat()
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    async def _save_report_metadata(self, metadata: ReportMetadata):
        """Save report metadata to storage"""
        metadata_file = self.data_directory / f"metadata_{metadata.generation_id}.json"
        
        metadata_dict = asdict(metadata)
        
        # Convert datetime objects
        for key, value in metadata_dict.items():
            if isinstance(value, datetime):
                metadata_dict[key] = value.isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
    
    def _start_scheduler(self):
        """Start the report generation scheduler"""
        
        def check_scheduled_reports():
            now = datetime.now(timezone.utc)
            
            for config in self.configurations.values():
                if config.is_active and config.next_generation <= now:
                    self.generation_queue.append(config.report_id)
        
        # Check every minute for scheduled reports
        schedule.every(1).minutes.do(check_scheduled_reports)
        
        # Start background task to process queue
        asyncio.create_task(self._process_generation_queue())
    
    async def _process_generation_queue(self):
        """Process the report generation queue"""
        while True:
            try:
                if self.generation_queue:
                    report_id = self.generation_queue.pop(0)
                    await self.generate_report(report_id)
                
                await asyncio.sleep(30)  # Check queue every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error processing generation queue: {str(e)}")
                await asyncio.sleep(60)
    
    def get_report_status(self, generation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific report generation"""
        
        if generation_id in self.report_metadata:
            metadata = self.report_metadata[generation_id]
            return {
                "generation_id": generation_id,
                "report_type": metadata.report_type.value,
                "status": "completed",
                "generated_at": metadata.generated_at.isoformat(),
                "file_path": metadata.file_path,
                "file_size": metadata.file_size_bytes,
                "distribution_status": metadata.distribution_status,
                "compliance_status": metadata.compliance_status
            }
        
        return None
    
    def list_active_configurations(self) -> List[Dict[str, Any]]:
        """List all active report configurations"""
        
        active_configs = []
        
        for config in self.configurations.values():
            if config.is_active:
                active_configs.append({
                    "report_id": config.report_id,
                    "name": config.name,
                    "report_type": config.report_type.value,
                    "frequency": config.frequency.value,
                    "next_generation": config.next_generation.isoformat(),
                    "jurisdiction": config.jurisdiction,
                    "regulatory_body": config.regulatory_body
                })
        
        return active_configs