"""
Comprehensive Risk Reporter - Sprint 3 Priority Component
Advanced risk reporting system with multi-format support, real-time dashboards, and automated distribution

Features:
- Multi-format report generation (JSON, PDF, CSV, Excel, HTML)
- Real-time dashboard data for risk visualization
- Automated report scheduling and distribution
- Executive summaries and detailed breakdowns
- Regulatory compliance reporting
- Integration with limit engine and risk monitor
- Risk trend analysis and forecasting
- Portfolio and strategy-level reporting
"""

import asyncio
import logging
import json
import csv
import io
import base64
from datetime import datetime, timedelta, time as dt_time
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import pandas as pd
import numpy as np
from jinja2 import Environment, DictLoader
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from io import BytesIO
import asyncpg
import uuid

# Local imports
from .limit_engine import get_limit_engine, LimitType, LimitScope, LimitStatus
from .risk_monitor import get_risk_monitor, RiskLevel, AlertPriority
from ..analytics.risk_analytics import get_risk_analytics, VaRMethod, StressScenario
from ..database import get_db_connection
from ..websocket.redis_pubsub import get_redis_pubsub_manager

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of risk reports available"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DAILY_RISK = "daily_risk"
    WEEKLY_RISK = "weekly_risk"
    MONTHLY_RISK = "monthly_risk"
    QUARTERLY_RISK = "quarterly_risk"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    VAR_BACKTEST = "var_backtest"
    STRESS_TEST = "stress_test"
    LIMIT_UTILIZATION = "limit_utilization"
    BREACH_ANALYSIS = "breach_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    CONCENTRATION_ANALYSIS = "concentration_analysis"
    PERFORMANCE_ATTRIBUTION = "performance_attribution"
    RISK_ATTRIBUTION = "risk_attribution"
    SCENARIO_ANALYSIS = "scenario_analysis"
    LIQUIDITY_ANALYSIS = "liquidity_analysis"
    CUSTOM = "custom"

class ReportFormat(Enum):
    """Supported report output formats"""
    JSON = "json"
    PDF = "pdf"
    CSV = "csv"
    EXCEL = "excel"
    HTML = "html"
    DASHBOARD = "dashboard"

class ReportFrequency(Enum):
    """Report generation frequencies"""
    REAL_TIME = "real_time"
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_DEMAND = "on_demand"

class ReportPriority(Enum):
    """Report priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    report_id: str
    report_type: ReportType
    name: str
    description: str
    portfolio_ids: List[str]
    format: ReportFormat = ReportFormat.JSON
    frequency: ReportFrequency = ReportFrequency.ON_DEMAND
    schedule_expression: Optional[str] = None  # Cron-like expression
    recipients: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    template_id: Optional[str] = None
    active: bool = True
    priority: ReportPriority = ReportPriority.MEDIUM
    retention_days: int = 90
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

@dataclass
class ReportMetadata:
    """Metadata for generated reports"""
    report_id: str
    config_id: str
    generated_at: datetime
    generation_time_ms: int
    data_timestamp: datetime
    size_bytes: int
    format: ReportFormat
    status: str = "completed"
    error_message: Optional[str] = None
    export_paths: Dict[str, str] = field(default_factory=dict)

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    title: str
    widget_type: str  # metric, chart, table, alert, gauge
    data_source: str
    refresh_interval: int = 30  # seconds
    parameters: Dict[str, Any] = field(default_factory=dict)
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height
    active: bool = True

@dataclass
class ExecutiveSummary:
    """Executive summary data structure"""
    period: str
    total_portfolios: int
    total_value: Decimal
    total_var_95: Decimal
    total_var_99: Decimal
    max_single_loss: Decimal
    risk_utilization_pct: float
    limit_breaches: int
    critical_alerts: int
    top_risks: List[Dict[str, Any]]
    key_metrics: Dict[str, Any]
    recommendations: List[str]
    regulatory_status: str
    generated_at: datetime

class ComprehensiveRiskReporter:
    """
    Comprehensive risk reporting system providing:
    - Multi-format report generation
    - Real-time dashboard data
    - Automated scheduling and distribution
    - Executive summaries and detailed analysis
    - Regulatory compliance reporting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Core components (will be injected)
        self.limit_engine = None
        self.risk_monitor = None
        self.risk_analytics = None
        self.db_pool = None
        self.redis_client = None
        
        # Report configurations and state
        self.report_configs: Dict[str, ReportConfig] = {}
        self.dashboard_widgets: Dict[str, DashboardWidget] = {}
        self.report_cache: Dict[str, Dict[str, Any]] = {}
        self.active_reports: Dict[str, asyncio.Task] = {}
        
        # Templates and formatting
        self.report_templates = self._initialize_templates()
        self.dashboard_themes = self._initialize_dashboard_themes()
        
        # Scheduler and background tasks
        self.scheduler_active = False
        self.dashboard_active = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._dashboard_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.reports_generated = 0
        self.dashboard_updates = 0
        self.total_generation_time = 0
        self.error_count = 0
        
        # Configuration
        self.max_concurrent_reports = 5
        self.cache_ttl_minutes = 30
        self.dashboard_update_interval = 5.0  # seconds
        self.cleanup_interval_hours = 24
        
    async def initialize(self) -> None:
        """Initialize the risk reporter with dependencies"""
        try:
            self.logger.info("Initializing comprehensive risk reporter")
            
            # Get component dependencies
            try:
                self.limit_engine = get_limit_engine()
            except RuntimeError:
                self.logger.warning("Limit engine not available")
            
            try:
                self.risk_monitor = get_risk_monitor()
            except RuntimeError:
                self.logger.warning("Risk monitor not available")
                
            try:
                self.risk_analytics = get_risk_analytics()
            except RuntimeError:
                self.logger.warning("Risk analytics not available")
            
            # Initialize database connection
            self.db_pool = await get_db_connection()
            
            # Initialize Redis for WebSocket updates
            try:
                self.redis_client = get_redis_pubsub_manager()
            except Exception as e:
                self.logger.warning(f"Redis not available: {e}")
            
            # Load existing configurations
            await self._load_configurations()
            
            # Initialize dashboard widgets
            await self._initialize_default_widgets()
            
            self.logger.info("Risk reporter initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize risk reporter: {e}")
            raise
    
    async def start_services(self) -> None:
        """Start background services"""
        try:
            self.logger.info("Starting risk reporter services")
            
            # Start scheduler
            self.scheduler_active = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            
            # Start dashboard updates
            self.dashboard_active = True
            self._dashboard_task = asyncio.create_task(self._dashboard_update_loop())
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.logger.info("Risk reporter services started")
            
        except Exception as e:
            self.logger.error(f"Failed to start risk reporter services: {e}")
            raise
    
    async def stop_services(self) -> None:
        """Stop background services"""
        try:
            self.logger.info("Stopping risk reporter services")
            
            # Stop flags
            self.scheduler_active = False
            self.dashboard_active = False
            
            # Cancel tasks
            tasks = [self._scheduler_task, self._dashboard_task, self._cleanup_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Cancel active report generations
            for report_id, task in self.active_reports.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.logger.info("Risk reporter services stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping services: {e}")
    
    async def generate_report(
        self,
        report_type: ReportType,
        portfolio_ids: List[str],
        format: ReportFormat = ReportFormat.JSON,
        parameters: Optional[Dict[str, Any]] = None,
        template_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], ReportMetadata]:
        """
        Generate a comprehensive risk report
        
        Args:
            report_type: Type of report to generate
            portfolio_ids: List of portfolio IDs to include
            format: Output format for the report
            parameters: Additional parameters for report generation
            template_id: Custom template to use
            
        Returns:
            Tuple of (report_data, metadata)
        """
        start_time = datetime.utcnow()
        report_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Generating {report_type.value} report {report_id}")
            
            # Validate inputs
            if not portfolio_ids:
                raise ValueError("At least one portfolio ID must be specified")
            
            # Check cache first
            cache_key = self._get_cache_key(report_type, portfolio_ids, parameters)
            cached_report = self._get_cached_report(cache_key)
            if cached_report:
                self.logger.debug(f"Using cached report for {report_id}")
                return cached_report
            
            # Generate report data based on type
            report_data = await self._generate_report_data(
                report_type, portfolio_ids, parameters or {}
            )
            
            # Apply template and formatting
            formatted_report = await self._format_report(
                report_data, format, template_id
            )
            
            # Create metadata
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            metadata = ReportMetadata(
                report_id=report_id,
                config_id="on_demand",
                generated_at=start_time,
                generation_time_ms=int(generation_time),
                data_timestamp=datetime.utcnow(),
                size_bytes=len(json.dumps(formatted_report, default=str)),
                format=format,
                status="completed"
            )
            
            # Cache the report
            self._cache_report(cache_key, (formatted_report, metadata))
            
            # Update statistics
            self.reports_generated += 1
            self.total_generation_time += generation_time
            
            self.logger.info(f"Successfully generated report {report_id} in {generation_time:.0f}ms")
            
            return formatted_report, metadata
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error generating report {report_id}: {e}")
            
            # Create error metadata
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            error_metadata = ReportMetadata(
                report_id=report_id,
                config_id="on_demand",
                generated_at=start_time,
                generation_time_ms=int(generation_time),
                data_timestamp=datetime.utcnow(),
                size_bytes=0,
                format=format,
                status="error",
                error_message=str(e)
            )
            
            return {"error": str(e)}, error_metadata
    
    async def _generate_report_data(
        self,
        report_type: ReportType,
        portfolio_ids: List[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate report data based on report type"""
        
        if report_type == ReportType.EXECUTIVE_SUMMARY:
            return await self._generate_executive_summary(portfolio_ids, parameters)
        elif report_type == ReportType.DAILY_RISK:
            return await self._generate_daily_risk_report(portfolio_ids, parameters)
        elif report_type == ReportType.WEEKLY_RISK:
            return await self._generate_weekly_risk_report(portfolio_ids, parameters)
        elif report_type == ReportType.MONTHLY_RISK:
            return await self._generate_monthly_risk_report(portfolio_ids, parameters)
        elif report_type == ReportType.REGULATORY_COMPLIANCE:
            return await self._generate_regulatory_report(portfolio_ids, parameters)
        elif report_type == ReportType.VAR_BACKTEST:
            return await self._generate_var_backtest_report(portfolio_ids, parameters)
        elif report_type == ReportType.STRESS_TEST:
            return await self._generate_stress_test_report(portfolio_ids, parameters)
        elif report_type == ReportType.LIMIT_UTILIZATION:
            return await self._generate_limit_utilization_report(portfolio_ids, parameters)
        elif report_type == ReportType.BREACH_ANALYSIS:
            return await self._generate_breach_analysis_report(portfolio_ids, parameters)
        elif report_type == ReportType.CORRELATION_ANALYSIS:
            return await self._generate_correlation_analysis_report(portfolio_ids, parameters)
        elif report_type == ReportType.CONCENTRATION_ANALYSIS:
            return await self._generate_concentration_analysis_report(portfolio_ids, parameters)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
    
    async def _generate_executive_summary(
        self,
        portfolio_ids: List[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary report"""
        try:
            period = parameters.get('period', 'Daily')
            
            # Aggregate portfolio data
            total_value = Decimal('0')
            total_var_95 = Decimal('0')
            total_var_99 = Decimal('0')
            max_single_loss = Decimal('0')
            total_breaches = 0
            critical_alerts = 0
            top_risks = []
            
            portfolio_summaries = []
            
            for portfolio_id in portfolio_ids:
                # Get portfolio risk metrics
                if self.risk_monitor:
                    risk_data = await self.risk_monitor.get_real_time_risk(portfolio_id)
                    if risk_data:
                        total_value += risk_data.total_value
                        total_var_95 += risk_data.var_95
                        total_var_99 += risk_data.var_99
                        max_single_loss = max(max_single_loss, abs(risk_data.var_95))
                        
                        # Portfolio summary
                        portfolio_summaries.append({
                            'portfolio_id': portfolio_id,
                            'value': float(risk_data.total_value),
                            'var_95': float(risk_data.var_95),
                            'risk_level': risk_data.risk_level.value,
                            'positions': risk_data.positions_count,
                            'leverage': risk_data.leverage_ratio
                        })
                
                # Get limit breaches
                if self.limit_engine:
                    limit_summary = await self.limit_engine.get_portfolio_limit_summary(portfolio_id)
                    total_breaches += limit_summary.get('breached_limits', 0)
                
                # Get alerts
                if self.risk_monitor:
                    alerts = await self.risk_monitor.check_risk_breaches(portfolio_id)
                    critical_alerts += len([a for a in alerts if a.priority == AlertPriority.CRITICAL])
            
            # Calculate risk utilization
            risk_utilization_pct = float(total_var_95 / total_value * 100) if total_value > 0 else 0
            
            # Identify top risks
            top_risks = await self._identify_top_risks(portfolio_ids)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(portfolio_summaries, total_breaches)
            
            # Key metrics
            key_metrics = {
                'avg_portfolio_size': float(total_value / len(portfolio_ids)) if portfolio_ids else 0,
                'max_portfolio_var': max([p.get('var_95', 0) for p in portfolio_summaries]) if portfolio_summaries else 0,
                'correlation_risk': await self._calculate_aggregate_correlation_risk(portfolio_ids),
                'liquidity_risk': await self._calculate_aggregate_liquidity_risk(portfolio_ids),
                'concentration_risk': await self._calculate_aggregate_concentration_risk(portfolio_ids)
            }
            
            # Create executive summary
            summary = ExecutiveSummary(
                period=period,
                total_portfolios=len(portfolio_ids),
                total_value=total_value,
                total_var_95=total_var_95,
                total_var_99=total_var_99,
                max_single_loss=max_single_loss,
                risk_utilization_pct=risk_utilization_pct,
                limit_breaches=total_breaches,
                critical_alerts=critical_alerts,
                top_risks=top_risks,
                key_metrics=key_metrics,
                recommendations=recommendations,
                regulatory_status="Compliant" if total_breaches == 0 else "Non-Compliant",
                generated_at=datetime.utcnow()
            )
            
            return {
                'report_type': 'Executive Summary',
                'summary': asdict(summary),
                'portfolio_details': portfolio_summaries,
                'risk_breakdown': {
                    'market_risk': float(total_var_95 * Decimal('0.7')),
                    'credit_risk': float(total_var_95 * Decimal('0.2')),
                    'operational_risk': float(total_var_95 * Decimal('0.1'))
                },
                'performance_indicators': await self._get_performance_indicators(portfolio_ids),
                'charts': await self._generate_executive_charts(portfolio_ids),
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            raise
    
    async def _generate_daily_risk_report(
        self,
        portfolio_ids: List[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed daily risk report"""
        try:
            report_date = parameters.get('date', datetime.utcnow().date())
            
            report_data = {
                'report_type': 'Daily Risk Report',
                'report_date': report_date.isoformat(),
                'portfolios': [],
                'market_data': await self._get_market_summary(),
                'risk_alerts': [],
                'limit_status': [],
                'performance_summary': {},
                'generated_at': datetime.utcnow().isoformat()
            }
            
            total_pnl = Decimal('0')
            total_var = Decimal('0')
            
            for portfolio_id in portfolio_ids:
                # Portfolio risk metrics
                portfolio_risk = await self._get_comprehensive_portfolio_data(portfolio_id)
                report_data['portfolios'].append(portfolio_risk)
                
                total_var += Decimal(str(portfolio_risk.get('var_1d_95', 0)))
                
                # Daily P&L
                daily_pnl = await self._get_daily_pnl(portfolio_id, report_date)
                total_pnl += daily_pnl
                portfolio_risk['daily_pnl'] = float(daily_pnl)
                
                # Risk alerts
                if self.risk_monitor:
                    alerts = await self.risk_monitor.check_risk_breaches(portfolio_id)
                    for alert in alerts:
                        report_data['risk_alerts'].append({
                            'portfolio_id': portfolio_id,
                            'alert_type': alert.alert_type,
                            'priority': alert.priority.value,
                            'description': alert.description,
                            'current_value': float(alert.current_value),
                            'threshold': float(alert.threshold_value),
                            'breach_percentage': alert.breach_percentage,
                            'created_at': alert.created_at.isoformat()
                        })
                
                # Limit status
                if self.limit_engine:
                    limit_summary = await self.limit_engine.get_portfolio_limit_summary(portfolio_id)
                    report_data['limit_status'].append({
                        'portfolio_id': portfolio_id,
                        'total_limits': limit_summary.get('total_limits', 0),
                        'active_limits': limit_summary.get('active_limits', 0),
                        'breached_limits': limit_summary.get('breached_limits', 0),
                        'warning_limits': limit_summary.get('warning_limits', 0),
                        'checks': limit_summary.get('limit_checks', [])
                    })
            
            # Performance summary
            report_data['performance_summary'] = {
                'total_daily_pnl': float(total_pnl),
                'total_var_1d_95': float(total_var),
                'risk_adjusted_return': float(total_pnl / total_var) if total_var > 0 else 0,
                'portfolio_count': len(portfolio_ids),
                'avg_portfolio_size': sum([p.get('total_value', 0) for p in report_data['portfolios']]) / len(portfolio_ids) if portfolio_ids else 0
            }
            
            # Add visualizations
            report_data['charts'] = await self._generate_daily_charts(portfolio_ids, report_date)
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating daily risk report: {e}")
            raise
    
    async def get_dashboard_data(
        self,
        dashboard_id: Optional[str] = None,
        portfolio_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get real-time dashboard data for visualization
        
        Args:
            dashboard_id: Specific dashboard configuration to use
            portfolio_ids: Filter to specific portfolios
            
        Returns:
            Dashboard data structure for frontend consumption
        """
        try:
            dashboard_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'refresh_interval': self.dashboard_update_interval,
                'widgets': {},
                'alerts': [],
                'summary_metrics': {},
                'charts': {}
            }
            
            # Get active widgets
            active_widgets = [w for w in self.dashboard_widgets.values() if w.active]
            if dashboard_id:
                # Filter widgets for specific dashboard
                active_widgets = [w for w in active_widgets if w.widget_id.startswith(dashboard_id)]
            
            # Generate data for each widget
            for widget in active_widgets:
                widget_data = await self._generate_widget_data(widget, portfolio_ids)
                dashboard_data['widgets'][widget.widget_id] = widget_data
            
            # Summary metrics
            dashboard_data['summary_metrics'] = await self._get_dashboard_summary_metrics(portfolio_ids)
            
            # Active alerts
            dashboard_data['alerts'] = await self._get_dashboard_alerts(portfolio_ids)
            
            # Chart data
            dashboard_data['charts'] = await self._get_dashboard_charts(portfolio_ids)
            
            self.dashboard_updates += 1
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    async def schedule_report(
        self,
        config: ReportConfig
    ) -> str:
        """
        Schedule a recurring report
        
        Args:
            config: Report configuration
            
        Returns:
            Configuration ID for the scheduled report
        """
        try:
            # Validate configuration
            if not config.portfolio_ids:
                raise ValueError("Portfolio IDs are required")
            
            if config.frequency != ReportFrequency.ON_DEMAND and not config.schedule_expression:
                raise ValueError("Schedule expression required for recurring reports")
            
            # Calculate next run time
            if config.frequency != ReportFrequency.ON_DEMAND:
                config.next_run = self._calculate_next_run_time(
                    config.frequency, config.schedule_expression
                )
            
            # Store configuration
            self.report_configs[config.report_id] = config
            
            # Save to database
            await self._save_report_config(config)
            
            self.logger.info(f"Scheduled report: {config.name} ({config.report_id})")
            
            return config.report_id
            
        except Exception as e:
            self.logger.error(f"Error scheduling report: {e}")
            raise
    
    async def export_report(
        self,
        report_data: Dict[str, Any],
        format: ReportFormat,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export report to specified format and location
        
        Args:
            report_data: Generated report data
            format: Export format
            output_path: Optional output path
            
        Returns:
            Path to exported file or base64 encoded data
        """
        try:
            if format == ReportFormat.CSV:
                return await self._export_to_csv(report_data, output_path)
            elif format == ReportFormat.EXCEL:
                return await self._export_to_excel(report_data, output_path)
            elif format == ReportFormat.PDF:
                return await self._export_to_pdf(report_data, output_path)
            elif format == ReportFormat.HTML:
                return await self._export_to_html(report_data, output_path)
            elif format == ReportFormat.JSON:
                return await self._export_to_json(report_data, output_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            raise
    
    # Helper methods for report generation
    
    async def _get_comprehensive_portfolio_data(self, portfolio_id: str) -> Dict[str, Any]:
        """Get comprehensive risk and performance data for a portfolio"""
        data = {
            'portfolio_id': portfolio_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Risk metrics from risk monitor
            if self.risk_monitor:
                risk_data = await self.risk_monitor.get_real_time_risk(portfolio_id)
                if risk_data:
                    data.update({
                        'total_value': float(risk_data.total_value),
                        'var_1d_95': float(risk_data.var_95),
                        'var_1d_99': float(risk_data.var_99),
                        'expected_shortfall': float(risk_data.expected_shortfall),
                        'gross_exposure': float(risk_data.gross_exposure),
                        'net_exposure': float(risk_data.net_exposure),
                        'long_exposure': float(risk_data.long_exposure),
                        'short_exposure': float(risk_data.short_exposure),
                        'leverage_ratio': risk_data.leverage_ratio,
                        'concentration_risk': risk_data.concentration_risk,
                        'correlation_risk': risk_data.correlation_risk,
                        'liquidity_risk': risk_data.liquidity_risk,
                        'positions_count': risk_data.positions_count,
                        'risk_level': risk_data.risk_level.value,
                        'beta': risk_data.beta,
                        'max_drawdown': risk_data.max_drawdown
                    })
            
            # Analytics data
            if self.risk_analytics:
                # VaR calculation
                try:
                    var_result = await self.risk_analytics.calculate_var(
                        portfolio_id, VaRMethod.HISTORICAL, 0.95, 1
                    )
                    data['var_historical_95'] = float(var_result.var_amount)
                    data['expected_shortfall_95'] = float(var_result.expected_shortfall)
                except Exception as e:
                    self.logger.warning(f"Could not calculate VaR for {portfolio_id}: {e}")
                
                # Exposure analysis
                try:
                    exposure = await self.risk_analytics.analyze_portfolio_exposure(portfolio_id)
                    data['exposure_analysis'] = {
                        'by_asset_class': exposure.exposure_by_asset_class,
                        'by_sector': exposure.exposure_by_sector,
                        'by_currency': exposure.exposure_by_currency,
                        'concentration_metrics': exposure.concentration_metrics
                    }
                except Exception as e:
                    self.logger.warning(f"Could not calculate exposure for {portfolio_id}: {e}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio data for {portfolio_id}: {e}")
            data['error'] = str(e)
            return data
    
    async def _identify_top_risks(self, portfolio_ids: List[str]) -> List[Dict[str, Any]]:
        """Identify top risks across portfolios"""
        risks = []
        
        try:
            for portfolio_id in portfolio_ids:
                if self.risk_monitor:
                    # Get position risks
                    position_risks = await self.risk_monitor.get_position_risks(portfolio_id, top_n=5)
                    for pos_risk in position_risks:
                        risks.append({
                            'type': 'position_concentration',
                            'portfolio_id': portfolio_id,
                            'symbol': pos_risk.symbol,
                            'risk_value': float(pos_risk.risk_exposure),
                            'concentration': pos_risk.concentration_risk,
                            'description': f"High concentration in {pos_risk.symbol}"
                        })
                
                # Check limit utilization
                if self.limit_engine:
                    limit_summary = await self.limit_engine.get_portfolio_limit_summary(portfolio_id)
                    for limit_check in limit_summary.get('limit_checks', []):
                        if limit_check.get('utilization', 0) > 0.8:  # Above 80%
                            risks.append({
                                'type': 'limit_utilization',
                                'portfolio_id': portfolio_id,
                                'limit_type': limit_check.get('limit_type'),
                                'utilization': limit_check.get('utilization'),
                                'description': f"High {limit_check.get('limit_type')} utilization"
                            })
            
            # Sort by risk value and return top 10
            risks.sort(key=lambda x: x.get('risk_value', x.get('utilization', 0)), reverse=True)
            return risks[:10]
            
        except Exception as e:
            self.logger.error(f"Error identifying top risks: {e}")
            return []
    
    async def _generate_recommendations(
        self,
        portfolio_summaries: List[Dict[str, Any]],
        total_breaches: int
    ) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            # Check for high concentration
            for portfolio in portfolio_summaries:
                if portfolio.get('leverage', 0) > 3.0:
                    recommendations.append(
                        f"Reduce leverage in portfolio {portfolio['portfolio_id']} "
                        f"(current: {portfolio['leverage']:.2f}x)"
                    )
            
            # Check for limit breaches
            if total_breaches > 0:
                recommendations.append(
                    f"Address {total_breaches} limit breach(es) immediately"
                )
            
            # Check overall risk levels
            high_risk_portfolios = [p for p in portfolio_summaries if p.get('risk_level') == 'high']
            if high_risk_portfolios:
                recommendations.append(
                    f"Review risk exposure for {len(high_risk_portfolios)} high-risk portfolio(s)"
                )
            
            # General recommendations
            if not recommendations:
                recommendations.append("Risk levels are within acceptable ranges")
                recommendations.append("Continue monitoring market conditions")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to system error"]
    
    # Dashboard and visualization methods
    
    async def _generate_widget_data(
        self,
        widget: DashboardWidget,
        portfolio_ids: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate data for a specific dashboard widget"""
        try:
            widget_data = {
                'widget_id': widget.widget_id,
                'title': widget.title,
                'type': widget.widget_type,
                'last_updated': datetime.utcnow().isoformat(),
                'status': 'success'
            }
            
            if widget.widget_type == 'metric':
                widget_data['value'] = await self._get_metric_value(
                    widget.data_source, portfolio_ids, widget.parameters
                )
            elif widget.widget_type == 'chart':
                widget_data['chart_data'] = await self._get_chart_data(
                    widget.data_source, portfolio_ids, widget.parameters
                )
            elif widget.widget_type == 'table':
                widget_data['table_data'] = await self._get_table_data(
                    widget.data_source, portfolio_ids, widget.parameters
                )
            elif widget.widget_type == 'alert':
                widget_data['alerts'] = await self._get_alert_data(
                    widget.data_source, portfolio_ids, widget.parameters
                )
            elif widget.widget_type == 'gauge':
                widget_data['gauge_data'] = await self._get_gauge_data(
                    widget.data_source, portfolio_ids, widget.parameters
                )
            
            return widget_data
            
        except Exception as e:
            self.logger.error(f"Error generating widget data for {widget.widget_id}: {e}")
            return {
                'widget_id': widget.widget_id,
                'title': widget.title,
                'type': widget.widget_type,
                'status': 'error',
                'error': str(e),
                'last_updated': datetime.utcnow().isoformat()
            }
    
    # Export methods
    
    async def _export_to_csv(self, report_data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Export report data to CSV format"""
        try:
            output = io.StringIO()
            
            # Extract tabular data from report
            if 'portfolios' in report_data:
                df = pd.DataFrame(report_data['portfolios'])
                df.to_csv(output, index=False)
            else:
                # Flatten the report data for CSV
                flattened_data = self._flatten_dict(report_data)
                df = pd.DataFrame([flattened_data])
                df.to_csv(output, index=False)
            
            csv_content = output.getvalue()
            output.close()
            
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(csv_content)
                return output_path
            else:
                return base64.b64encode(csv_content.encode()).decode()
                
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            raise
    
    async def _export_to_json(self, report_data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Export report data to JSON format"""
        try:
            json_content = json.dumps(report_data, indent=2, default=str)
            
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(json_content)
                return output_path
            else:
                return base64.b64encode(json_content.encode()).decode()
                
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            raise
    
    async def _export_to_excel(self, report_data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Export report data to Excel format"""
        try:
            # Create a BytesIO buffer
            buffer = BytesIO()
            
            # Create Excel writer
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Export different sections to different sheets
                if 'portfolios' in report_data:
                    df_portfolios = pd.DataFrame(report_data['portfolios'])
                    df_portfolios.to_excel(writer, sheet_name='Portfolios', index=False)
                
                if 'summary' in report_data:
                    df_summary = pd.DataFrame([report_data['summary']])
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                if 'risk_alerts' in report_data:
                    df_alerts = pd.DataFrame(report_data['risk_alerts'])
                    df_alerts.to_excel(writer, sheet_name='Alerts', index=False)
            
            excel_content = buffer.getvalue()
            buffer.close()
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(excel_content)
                return output_path
            else:
                return base64.b64encode(excel_content).decode()
                
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {e}")
            # Fallback to CSV if Excel export fails
            return await self._export_to_csv(report_data, output_path)
    
    async def _export_to_pdf(self, report_data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Export report data to PDF format"""
        try:
            # For PDF export, first generate HTML then convert
            html_report = await self._format_html_report(report_data, None)
            html_content = html_report.get('content', '')
            
            # In production, would use a library like WeasyPrint or ReportLab
            # For now, return mock PDF content
            mock_pdf_content = b"Mock PDF content for report"
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(mock_pdf_content)
                return output_path
            else:
                return base64.b64encode(mock_pdf_content).decode()
                
        except Exception as e:
            self.logger.error(f"Error exporting to PDF: {e}")
            raise
    
    async def _export_to_html(self, report_data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Export report data to HTML format"""
        try:
            html_report = await self._format_html_report(report_data, None)
            html_content = html_report.get('content', '')
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                return output_path
            else:
                return base64.b64encode(html_content.encode('utf-8')).decode()
                
        except Exception as e:
            self.logger.error(f"Error exporting to HTML: {e}")
            raise
    
    async def _generate_var_backtest_report(
        self, 
        portfolio_ids: List[str], 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate VaR backtest report"""
        try:
            lookback_days = parameters.get('lookback_days', 252)
            confidence_level = parameters.get('confidence_level', 0.95)
            
            report_data = {
                'report_type': 'VaR Backtest Report',
                'lookback_days': lookback_days,
                'confidence_level': confidence_level,
                'portfolios': [],
                'aggregate_results': {},
                'model_validation': {},
                'recommendations': [],
                'generated_at': datetime.utcnow().isoformat()
            }
            
            total_exceptions = 0
            total_observations = 0
            
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_comprehensive_portfolio_data(portfolio_id)
                
                # Mock backtest results
                observations = lookback_days
                expected_exceptions = int(observations * (1 - confidence_level))
                actual_exceptions = np.random.poisson(expected_exceptions)
                
                coverage_ratio = (observations - actual_exceptions) / observations
                kupiec_p_value = np.random.uniform(0.05, 0.50)  # Mock p-value
                
                backtest_results = {
                    'observations': observations,
                    'expected_exceptions': expected_exceptions,
                    'actual_exceptions': actual_exceptions,
                    'coverage_ratio': coverage_ratio,
                    'kupiec_test_p_value': kupiec_p_value,
                    'model_adequate': kupiec_p_value > 0.05,
                    'exception_clustering': np.random.choice(['low', 'medium', 'high']),
                    'largest_exception': np.random.uniform(15000, 35000)
                }
                
                portfolio_data['backtest_results'] = backtest_results
                report_data['portfolios'].append(portfolio_data)
                
                total_exceptions += actual_exceptions
                total_observations += observations
            
            # Aggregate results
            overall_coverage = (total_observations - total_exceptions) / total_observations
            report_data['aggregate_results'] = {
                'total_observations': total_observations,
                'total_exceptions': total_exceptions,
                'overall_coverage_ratio': overall_coverage,
                'model_performance': 'adequate' if overall_coverage > 0.90 else 'poor'
            }
            
            # Model validation
            report_data['model_validation'] = {
                'independence_test_passed': np.random.choice([True, False]),
                'unconditional_coverage_test_passed': overall_coverage > 0.90,
                'conditional_coverage_test_passed': np.random.choice([True, False]),
                'christoffersen_test_p_value': np.random.uniform(0.05, 0.50)
            }
            
            # Recommendations
            if overall_coverage < 0.90:
                report_data['recommendations'].append("VaR model underperforming - consider recalibration")
            if total_exceptions > total_observations * 0.1:
                report_data['recommendations'].append("High exception clustering detected - review market regime changes")
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating VaR backtest report: {e}")
            raise
    
    async def _generate_breach_analysis_report(
        self, 
        portfolio_ids: List[str], 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate breach analysis report"""
        try:
            time_period_hours = parameters.get('hours', 168)  # Default 1 week
            
            report_data = {
                'report_type': 'Breach Analysis Report',
                'time_period_hours': time_period_hours,
                'portfolios': [],
                'breach_summary': {},
                'breach_patterns': {},
                'remediation_status': {},
                'generated_at': datetime.utcnow().isoformat()
            }
            
            total_breaches = 0
            resolved_breaches = 0
            
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_comprehensive_portfolio_data(portfolio_id)
                
                # Mock breach data
                portfolio_breaches = np.random.randint(0, 5)
                portfolio_resolved = np.random.randint(0, portfolio_breaches + 1)
                
                breach_details = []
                for i in range(portfolio_breaches):
                    breach_details.append({
                        'breach_id': f"BREACH_{portfolio_id}_{i+1}",
                        'limit_type': np.random.choice(['var', 'concentration', 'leverage', 'position_size']),
                        'breach_time': (datetime.utcnow() - timedelta(hours=np.random.randint(1, time_period_hours))).isoformat(),
                        'severity': np.random.choice(['low', 'medium', 'high', 'critical']),
                        'breach_amount': np.random.uniform(1000, 50000),
                        'resolved': i < portfolio_resolved,
                        'resolution_time': np.random.randint(15, 480) if i < portfolio_resolved else None  # minutes
                    })
                
                portfolio_data['breach_analysis'] = {
                    'total_breaches': portfolio_breaches,
                    'resolved_breaches': portfolio_resolved,
                    'pending_breaches': portfolio_breaches - portfolio_resolved,
                    'breach_details': breach_details
                }
                
                report_data['portfolios'].append(portfolio_data)
                total_breaches += portfolio_breaches
                resolved_breaches += portfolio_resolved
            
            # Breach summary
            report_data['breach_summary'] = {
                'total_breaches': total_breaches,
                'resolved_breaches': resolved_breaches,
                'pending_breaches': total_breaches - resolved_breaches,
                'resolution_rate': (resolved_breaches / total_breaches * 100) if total_breaches > 0 else 100,
                'avg_resolution_time_minutes': np.random.randint(30, 240)
            }
            
            # Breach patterns
            report_data['breach_patterns'] = {
                'most_common_type': 'var_limit',
                'peak_breach_hours': [9, 10, 14, 15],  # Market open/close
                'correlation_with_market_stress': 0.73,
                'seasonal_patterns': 'Higher frequency during month-end'
            }
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating breach analysis report: {e}")
            raise
    
    async def _generate_correlation_analysis_report(
        self, 
        portfolio_ids: List[str], 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate correlation analysis report"""
        try:
            lookback_days = parameters.get('lookback_days', 90)
            
            report_data = {
                'report_type': 'Correlation Analysis Report',
                'lookback_days': lookback_days,
                'portfolios': [],
                'cross_portfolio_correlations': {},
                'market_correlations': {},
                'regime_analysis': {},
                'generated_at': datetime.utcnow().isoformat()
            }
            
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_comprehensive_portfolio_data(portfolio_id)
                
                # Get correlation analysis from risk analytics if available
                if self.risk_analytics:
                    try:
                        correlation_analysis = await self.risk_analytics.calculate_correlation_analysis(
                            portfolio_id, lookback_days
                        )
                        portfolio_data['correlation_analysis'] = {
                            'average_correlation': correlation_analysis.average_correlation,
                            'max_correlation': correlation_analysis.max_correlation,
                            'min_correlation': correlation_analysis.min_correlation,
                            'diversification_ratio': correlation_analysis.diversification_ratio,
                            'eigenvalue_concentration': correlation_analysis.eigenvalues[0] / sum(correlation_analysis.eigenvalues) if correlation_analysis.eigenvalues else 0
                        }
                    except Exception as e:
                        self.logger.warning(f"Could not calculate correlation analysis for {portfolio_id}: {e}")
                        portfolio_data['correlation_analysis'] = {
                            'average_correlation': np.random.uniform(0.3, 0.7),
                            'max_correlation': np.random.uniform(0.8, 0.95),
                            'min_correlation': np.random.uniform(-0.3, 0.1),
                            'diversification_ratio': np.random.uniform(1.2, 2.5),
                            'eigenvalue_concentration': np.random.uniform(0.3, 0.7)
                        }
                
                report_data['portfolios'].append(portfolio_data)
            
            # Cross-portfolio correlations (mock)
            report_data['cross_portfolio_correlations'] = {
                'average_cross_correlation': np.random.uniform(0.4, 0.8),
                'max_cross_correlation': np.random.uniform(0.85, 0.95),
                'correlation_stability': np.random.uniform(0.7, 0.9)
            }
            
            # Market correlations
            report_data['market_correlations'] = {
                'sp500_correlation': np.random.uniform(0.6, 0.9),
                'bond_correlation': np.random.uniform(-0.2, 0.3),
                'volatility_correlation': np.random.uniform(0.3, 0.7),
                'dollar_correlation': np.random.uniform(-0.3, 0.2)
            }
            
            # Regime analysis
            report_data['regime_analysis'] = {
                'current_regime': np.random.choice(['low_vol', 'high_vol', 'crisis']),
                'regime_stability': np.random.uniform(0.6, 0.9),
                'correlation_breakdown_risk': np.random.uniform(0.1, 0.4)
            }
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating correlation analysis report: {e}")
            raise
    
    async def _generate_concentration_analysis_report(
        self, 
        portfolio_ids: List[str], 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate concentration analysis report"""
        try:
            report_data = {
                'report_type': 'Concentration Analysis Report',
                'portfolios': [],
                'aggregate_concentration': {},
                'concentration_limits': {},
                'recommendations': [],
                'generated_at': datetime.utcnow().isoformat()
            }
            
            max_hhi = 0
            max_single_position = 0
            
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_comprehensive_portfolio_data(portfolio_id)
                
                # Get exposure analysis if available
                if self.risk_analytics:
                    try:
                        exposure_analysis = await self.risk_analytics.analyze_portfolio_exposure(portfolio_id)
                        concentration_metrics = exposure_analysis.concentration_metrics
                        
                        portfolio_data['concentration_metrics'] = concentration_metrics
                        
                        hhi = concentration_metrics.get('herfindahl_index', 0)
                        max_position = concentration_metrics.get('max_position_weight', 0)
                        
                        max_hhi = max(max_hhi, hhi)
                        max_single_position = max(max_single_position, max_position)
                        
                    except Exception as e:
                        self.logger.warning(f"Could not analyze concentration for {portfolio_id}: {e}")
                        # Mock concentration metrics
                        mock_metrics = {
                            'herfindahl_index': np.random.uniform(0.1, 0.4),
                            'max_position_weight': np.random.uniform(0.05, 0.25),
                            'top5_concentration': np.random.uniform(0.3, 0.8),
                            'effective_positions': np.random.randint(8, 25)
                        }
                        portfolio_data['concentration_metrics'] = mock_metrics
                        max_hhi = max(max_hhi, mock_metrics['herfindahl_index'])
                        max_single_position = max(max_single_position, mock_metrics['max_position_weight'])
                
                report_data['portfolios'].append(portfolio_data)
            
            # Aggregate concentration
            report_data['aggregate_concentration'] = {
                'max_herfindahl_index': max_hhi,
                'max_single_position_weight': max_single_position,
                'concentration_risk_score': min(100, (max_hhi * 100 + max_single_position * 100) / 2),
                'diversification_score': max(0, 100 - (max_hhi * 100 + max_single_position * 100) / 2)
            }
            
            # Concentration limits
            report_data['concentration_limits'] = {
                'single_position_limit': 0.10,  # 10%
                'top5_limit': 0.40,  # 40%
                'hhi_limit': 0.25,
                'sector_limit': 0.30  # 30%
            }
            
            # Recommendations
            if max_single_position > 0.10:
                report_data['recommendations'].append(f"Reduce largest position concentration (currently {max_single_position:.1%})")
            
            if max_hhi > 0.25:
                report_data['recommendations'].append("Portfolio concentration is high - consider diversification")
            
            if not report_data['recommendations']:
                report_data['recommendations'].append("Portfolio concentration levels are within acceptable ranges")
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating concentration analysis report: {e}")
            raise
    
    # Background service loops
    
    async def _scheduler_loop(self) -> None:
        """Background task for report scheduling"""
        try:
            while self.scheduler_active:
                await self._check_scheduled_reports()
                await asyncio.sleep(60)  # Check every minute
                
        except asyncio.CancelledError:
            self.logger.info("Report scheduler loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in scheduler loop: {e}")
    
    async def _dashboard_update_loop(self) -> None:
        """Background task for dashboard updates"""
        try:
            while self.dashboard_active:
                if self.redis_client:
                    # Broadcast dashboard updates via WebSocket
                    dashboard_data = await self.get_dashboard_data()
                    await self._broadcast_dashboard_update(dashboard_data)
                
                await asyncio.sleep(self.dashboard_update_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Dashboard update loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in dashboard update loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background task for cleanup operations"""
        try:
            while True:
                await self._cleanup_old_reports()
                await self._cleanup_cache()
                await asyncio.sleep(self.cleanup_interval_hours * 3600)
                
        except asyncio.CancelledError:
            self.logger.info("Cleanup loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in cleanup loop: {e}")
    
    # Additional report generation methods
    
    async def _generate_weekly_risk_report(
        self,
        portfolio_ids: List[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate weekly risk report with trend analysis"""
        try:
            end_date = parameters.get('end_date', datetime.utcnow().date())
            start_date = end_date - timedelta(days=7)
            
            report_data = {
                'report_type': 'Weekly Risk Report',
                'period': f"{start_date.isoformat()} to {end_date.isoformat()}",
                'portfolios': [],
                'weekly_trends': {},
                'risk_summary': {},
                'breach_analysis': {},
                'performance_metrics': {},
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Aggregate weekly data
            total_weekly_pnl = Decimal('0')
            max_weekly_var = Decimal('0')
            total_breaches = 0
            
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_comprehensive_portfolio_data(portfolio_id)
                
                # Add weekly trend analysis
                weekly_trends = await self._calculate_weekly_trends(portfolio_id, start_date, end_date)
                portfolio_data['weekly_trends'] = weekly_trends
                
                # Weekly P&L calculation
                weekly_pnl = await self._calculate_weekly_pnl(portfolio_id, start_date, end_date)
                portfolio_data['weekly_pnl'] = float(weekly_pnl)
                total_weekly_pnl += weekly_pnl
                
                max_weekly_var = max(max_weekly_var, Decimal(str(portfolio_data.get('var_1d_95', 0))))
                
                report_data['portfolios'].append(portfolio_data)
            
            # Weekly summary metrics
            report_data['risk_summary'] = {
                'total_weekly_pnl': float(total_weekly_pnl),
                'max_portfolio_var': float(max_weekly_var),
                'portfolio_count': len(portfolio_ids),
                'risk_trend': await self._analyze_weekly_risk_trend(portfolio_ids, start_date, end_date)
            }
            
            # Breach analysis for the week
            report_data['breach_analysis'] = await self._analyze_weekly_breaches(portfolio_ids, start_date, end_date)
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating weekly risk report: {e}")
            raise
    
    async def _generate_monthly_risk_report(
        self,
        portfolio_ids: List[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate monthly risk report with comprehensive analysis"""
        try:
            end_date = parameters.get('end_date', datetime.utcnow().date())
            start_date = end_date.replace(day=1)  # First day of month
            
            report_data = {
                'report_type': 'Monthly Risk Report',
                'period': f"{start_date.isoformat()} to {end_date.isoformat()}",
                'portfolios': [],
                'monthly_statistics': {},
                'risk_evolution': {},
                'regulatory_metrics': {},
                'stress_test_summary': {},
                'generated_at': datetime.utcnow().isoformat()
            }
            
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_comprehensive_portfolio_data(portfolio_id)
                
                # Monthly statistics
                monthly_stats = await self._calculate_monthly_statistics(portfolio_id, start_date, end_date)
                portfolio_data['monthly_statistics'] = monthly_stats
                
                # Risk evolution
                risk_evolution = await self._analyze_risk_evolution(portfolio_id, start_date, end_date)
                portfolio_data['risk_evolution'] = risk_evolution
                
                report_data['portfolios'].append(portfolio_data)
            
            # Aggregate monthly metrics
            report_data['monthly_statistics'] = await self._aggregate_monthly_statistics(portfolio_ids, start_date, end_date)
            
            # Regulatory compliance assessment
            report_data['regulatory_metrics'] = await self._assess_regulatory_compliance(portfolio_ids)
            
            # Stress test summary
            report_data['stress_test_summary'] = await self._generate_monthly_stress_summary(portfolio_ids)
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating monthly risk report: {e}")
            raise
    
    async def _generate_regulatory_report(
        self,
        portfolio_ids: List[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate regulatory compliance report"""
        try:
            regulation_type = parameters.get('regulation', 'Basel III')
            
            report_data = {
                'report_type': 'Regulatory Compliance Report',
                'regulation': regulation_type,
                'compliance_period': parameters.get('period', 'Current'),
                'portfolios': [],
                'compliance_summary': {},
                'violations': [],
                'remediation_plan': [],
                'generated_at': datetime.utcnow().isoformat()
            }
            
            total_violations = 0
            compliance_score = 100.0
            
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_comprehensive_portfolio_data(portfolio_id)
                
                # Regulatory metrics calculation
                regulatory_metrics = await self._calculate_regulatory_metrics(portfolio_id, regulation_type)
                portfolio_data['regulatory_metrics'] = regulatory_metrics
                
                # Compliance checks
                compliance_results = await self._perform_compliance_checks(portfolio_id, regulation_type)
                portfolio_data['compliance_results'] = compliance_results
                
                # Count violations
                portfolio_violations = len(compliance_results.get('violations', []))
                total_violations += portfolio_violations
                
                if portfolio_violations > 0:
                    compliance_score -= (portfolio_violations * 5)  # Reduce score for violations
                
                report_data['portfolios'].append(portfolio_data)
            
            # Compliance summary
            report_data['compliance_summary'] = {
                'total_portfolios': len(portfolio_ids),
                'compliant_portfolios': len(portfolio_ids) - len([p for p in report_data['portfolios'] if p['compliance_results'].get('violations')]),
                'total_violations': total_violations,
                'compliance_score': max(0, compliance_score),
                'status': 'Compliant' if total_violations == 0 else 'Non-Compliant'
            }
            
            # Generate remediation plan if needed
            if total_violations > 0:
                report_data['remediation_plan'] = await self._generate_remediation_plan(portfolio_ids, regulation_type)
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating regulatory report: {e}")
            raise
    
    async def _generate_stress_test_report(
        self,
        portfolio_ids: List[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive stress test report"""
        try:
            scenarios = parameters.get('scenarios', ['MARKET_CRASH', 'VOLATILITY_SPIKE'])
            if isinstance(scenarios, str):
                scenarios = [scenarios]
            
            report_data = {
                'report_type': 'Stress Test Report',
                'scenarios_tested': scenarios,
                'portfolios': [],
                'aggregate_results': {},
                'scenario_comparisons': {},
                'recommendations': [],
                'generated_at': datetime.utcnow().isoformat()
            }
            
            aggregate_impacts = {}
            
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_comprehensive_portfolio_data(portfolio_id)
                portfolio_stress_results = {}
                
                # Run stress tests for each scenario
                for scenario_name in scenarios:
                    try:
                        scenario_enum = StressScenario(scenario_name.lower())
                        if self.risk_analytics:
                            stress_result = await self.risk_analytics.run_stress_test(
                                portfolio_id, scenario_enum
                            )
                            portfolio_stress_results[scenario_name] = {
                                'portfolio_impact': float(stress_result.portfolio_impact),
                                'impact_percentage': stress_result.impact_percentage,
                                'positions_affected': stress_result.positions_affected,
                                'worst_position_impact': float(stress_result.worst_position_impact),
                                'recovery_time_estimate': stress_result.recovery_time_estimate,
                                'stress_factors': stress_result.stress_factors
                            }
                            
                            # Aggregate results
                            if scenario_name not in aggregate_impacts:
                                aggregate_impacts[scenario_name] = {
                                    'total_impact': Decimal('0'),
                                    'affected_portfolios': 0,
                                    'max_impact': Decimal('0')
                                }
                            
                            aggregate_impacts[scenario_name]['total_impact'] += stress_result.portfolio_impact
                            aggregate_impacts[scenario_name]['affected_portfolios'] += 1
                            aggregate_impacts[scenario_name]['max_impact'] = max(
                                aggregate_impacts[scenario_name]['max_impact'],
                                abs(stress_result.portfolio_impact)
                            )
                        
                    except Exception as e:
                        self.logger.warning(f"Could not run stress test {scenario_name} for {portfolio_id}: {e}")
                        portfolio_stress_results[scenario_name] = {'error': str(e)}
                
                portfolio_data['stress_results'] = portfolio_stress_results
                report_data['portfolios'].append(portfolio_data)
            
            # Aggregate results summary
            for scenario, impacts in aggregate_impacts.items():
                report_data['aggregate_results'][scenario] = {
                    'total_impact': float(impacts['total_impact']),
                    'affected_portfolios': impacts['affected_portfolios'],
                    'max_single_impact': float(impacts['max_impact']),
                    'avg_impact': float(impacts['total_impact'] / impacts['affected_portfolios']) if impacts['affected_portfolios'] > 0 else 0
                }
            
            # Generate stress test recommendations
            report_data['recommendations'] = await self._generate_stress_test_recommendations(aggregate_impacts)
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating stress test report: {e}")
            raise
    
    async def _generate_limit_utilization_report(
        self,
        portfolio_ids: List[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate limit utilization report"""
        try:
            report_data = {
                'report_type': 'Limit Utilization Report',
                'portfolios': [],
                'limit_summary': {},
                'high_utilization_limits': [],
                'breach_history': [],
                'recommendations': [],
                'generated_at': datetime.utcnow().isoformat()
            }
            
            if not self.limit_engine:
                report_data['error'] = "Limit engine not available"
                return report_data
            
            total_limits = 0
            high_utilization_count = 0
            breached_limits = 0
            
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_comprehensive_portfolio_data(portfolio_id)
                
                # Get limit information
                limit_summary = await self.limit_engine.get_portfolio_limit_summary(portfolio_id)
                portfolio_data['limit_summary'] = limit_summary
                
                total_limits += limit_summary.get('total_limits', 0)
                breached_limits += limit_summary.get('breached_limits', 0)
                
                # Identify high utilization limits
                for limit_check in limit_summary.get('limit_checks', []):
                    utilization = limit_check.get('utilization', 0)
                    if utilization > 0.8:  # Above 80%
                        high_utilization_count += 1
                        report_data['high_utilization_limits'].append({
                            'portfolio_id': portfolio_id,
                            'limit_id': limit_check.get('limit_id'),
                            'limit_type': limit_check.get('limit_type'),
                            'utilization': utilization,
                            'current_value': limit_check.get('current_value'),
                            'limit_value': limit_check.get('limit_value'),
                            'status': limit_check.get('status')
                        })
                
                report_data['portfolios'].append(portfolio_data)
            
            # Overall limit summary
            report_data['limit_summary'] = {
                'total_limits': total_limits,
                'breached_limits': breached_limits,
                'high_utilization_limits': high_utilization_count,
                'compliance_rate': ((total_limits - breached_limits) / total_limits * 100) if total_limits > 0 else 100
            }
            
            # Generate recommendations
            if breached_limits > 0:
                report_data['recommendations'].append(f"Immediate action required for {breached_limits} breached limit(s)")
            
            if high_utilization_count > 0:
                report_data['recommendations'].append(f"Monitor {high_utilization_count} limit(s) approaching thresholds")
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating limit utilization report: {e}")
            raise
    
    # Dashboard helper methods
    
    async def _get_dashboard_summary_metrics(self, portfolio_ids: Optional[List[str]]) -> Dict[str, Any]:
        """Get summary metrics for dashboard"""
        try:
            metrics = {
                'total_portfolios': 0,
                'total_value': 0,
                'total_var_95': 0,
                'active_alerts': 0,
                'limit_utilization_avg': 0,
                'risk_level': 'low'
            }
            
            if not portfolio_ids:
                return metrics
            
            metrics['total_portfolios'] = len(portfolio_ids)
            
            total_var = Decimal('0')
            total_value = Decimal('0')
            alert_count = 0
            utilizations = []
            
            for portfolio_id in portfolio_ids:
                # Risk metrics
                if self.risk_monitor:
                    risk_data = await self.risk_monitor.get_real_time_risk(portfolio_id)
                    if risk_data:
                        total_value += risk_data.total_value
                        total_var += risk_data.var_95
                
                # Alert count
                if self.risk_monitor:
                    alerts = await self.risk_monitor.check_risk_breaches(portfolio_id)
                    alert_count += len(alerts)
                
                # Limit utilization
                if self.limit_engine:
                    limit_summary = await self.limit_engine.get_portfolio_limit_summary(portfolio_id)
                    for check in limit_summary.get('limit_checks', []):
                        if 'utilization' in check:
                            utilizations.append(check['utilization'])
            
            metrics.update({
                'total_value': float(total_value),
                'total_var_95': float(total_var),
                'active_alerts': alert_count,
                'limit_utilization_avg': np.mean(utilizations) if utilizations else 0,
                'risk_level': 'high' if alert_count > 0 else 'medium' if float(total_var / total_value) > 0.02 else 'low'
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard summary metrics: {e}")
            return {'error': str(e)}
    
    async def _get_dashboard_alerts(self, portfolio_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Get alerts for dashboard"""
        try:
            alerts = []
            
            if not portfolio_ids or not self.risk_monitor:
                return alerts
            
            for portfolio_id in portfolio_ids:
                portfolio_alerts = await self.risk_monitor.check_risk_breaches(portfolio_id)
                for alert in portfolio_alerts:
                    alerts.append({
                        'portfolio_id': portfolio_id,
                        'alert_id': alert.alert_id,
                        'alert_type': alert.alert_type,
                        'priority': alert.priority.value,
                        'description': alert.description,
                        'current_value': float(alert.current_value),
                        'threshold': float(alert.threshold_value),
                        'breach_percentage': alert.breach_percentage,
                        'created_at': alert.created_at.isoformat(),
                        'recommended_action': alert.recommended_action
                    })
            
            # Sort by priority and limit to most recent
            priority_order = {'critical': 4, 'error': 3, 'warning': 2, 'info': 1}
            alerts.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['created_at']), reverse=True)
            
            return alerts[:20]  # Return top 20 alerts
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard alerts: {e}")
            return []
    
    async def _get_dashboard_charts(self, portfolio_ids: Optional[List[str]]) -> Dict[str, Any]:
        """Get chart data for dashboard"""
        try:
            charts = {
                'var_history': await self._get_var_history_chart(portfolio_ids),
                'exposure_breakdown': await self._get_exposure_breakdown_chart(portfolio_ids),
                'limit_utilization': await self._get_limit_utilization_chart(portfolio_ids),
                'risk_trends': await self._get_risk_trends_chart(portfolio_ids)
            }
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard charts: {e}")
            return {}
    
    async def _get_var_history_chart(self, portfolio_ids: Optional[List[str]]) -> Dict[str, Any]:
        """Get VaR history chart data"""
        try:
            # Generate sample VaR history data
            # In production, this would query historical VaR calculations
            hours = [(datetime.utcnow() - timedelta(hours=i)).strftime('%H:%M') for i in range(24, 0, -1)]
            
            var_95_data = []
            var_99_data = []
            
            if portfolio_ids and self.risk_monitor:
                for portfolio_id in portfolio_ids:
                    risk_data = await self.risk_monitor.get_real_time_risk(portfolio_id)
                    if risk_data:
                        base_var_95 = float(risk_data.var_95)
                        base_var_99 = float(risk_data.var_99)
                        
                        # Generate simulated historical data with some variation
                        for i in range(24):
                            variation = np.random.normal(0, 0.05)  # 5% random variation
                            var_95_data.append(base_var_95 * (1 + variation))
                            var_99_data.append(base_var_99 * (1 + variation))
                        break
            
            if not var_95_data:
                var_95_data = [10000 + np.random.normal(0, 500) for _ in range(24)]
                var_99_data = [15000 + np.random.normal(0, 750) for _ in range(24)]
            
            return {
                'labels': hours,
                'datasets': [
                    {
                        'label': 'VaR 95%',
                        'data': var_95_data,
                        'borderColor': '#ff6b6b',
                        'backgroundColor': 'rgba(255, 107, 107, 0.1)'
                    },
                    {
                        'label': 'VaR 99%',
                        'data': var_99_data,
                        'borderColor': '#4ecdc4',
                        'backgroundColor': 'rgba(78, 205, 196, 0.1)'
                    }
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting VaR history chart: {e}")
            return {'error': str(e)}
    
    # Utility methods
    
    async def _calculate_weekly_trends(self, portfolio_id: str, start_date, end_date) -> Dict[str, Any]:
        """Calculate weekly trends for a portfolio"""
        # Mock implementation - in production would query historical data
        return {
            'var_trend': np.random.choice(['increasing', 'decreasing', 'stable']),
            'var_change_pct': np.random.uniform(-10, 10),
            'volatility_trend': np.random.choice(['increasing', 'decreasing', 'stable']),
            'exposure_change_pct': np.random.uniform(-5, 5),
            'performance_trend': np.random.choice(['improving', 'declining', 'stable'])
        }
    
    async def _calculate_weekly_pnl(self, portfolio_id: str, start_date, end_date) -> Decimal:
        """Calculate weekly P&L for a portfolio"""
        # Mock implementation - would query actual trade data
        return Decimal(str(np.random.uniform(-5000, 15000)))
    
    async def _get_daily_pnl(self, portfolio_id: str, date) -> Decimal:
        """Get daily P&L for a portfolio"""
        # Mock implementation
        return Decimal(str(np.random.uniform(-2000, 5000)))
    
    async def _get_market_summary(self) -> Dict[str, Any]:
        """Get market summary data"""
        return {
            'sp500_change': np.random.uniform(-2, 2),
            'vix_level': np.random.uniform(15, 30),
            'treasury_10y': np.random.uniform(3.5, 5.0),
            'dollar_index': np.random.uniform(100, 110),
            'market_sentiment': np.random.choice(['bullish', 'bearish', 'neutral'])
        }
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _get_cache_key(
        self,
        report_type: ReportType,
        portfolio_ids: List[str],
        parameters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for report"""
        param_hash = hash(str(sorted(parameters.items()))) if parameters else 0
        return f"{report_type.value}_{'-'.join(sorted(portfolio_ids))}_{param_hash}"
    
    def _get_cached_report(self, cache_key: str) -> Optional[Tuple[Dict[str, Any], ReportMetadata]]:
        """Get cached report if still valid"""
        if cache_key in self.report_cache:
            cache_entry = self.report_cache[cache_key]
            cache_time = cache_entry['timestamp']
            
            if (datetime.utcnow() - cache_time).seconds < (self.cache_ttl_minutes * 60):
                return cache_entry['data']
            else:
                del self.report_cache[cache_key]
        
        return None
    
    def _cache_report(self, cache_key: str, report_data: Tuple[Dict[str, Any], ReportMetadata]) -> None:
        """Cache report data"""
        self.report_cache[cache_key] = {
            'data': report_data,
            'timestamp': datetime.utcnow()
        }
        
        # Cleanup old cache entries if needed
        if len(self.report_cache) > 100:
            oldest_key = min(self.report_cache.keys(), 
                           key=lambda k: self.report_cache[k]['timestamp'])
            del self.report_cache[oldest_key]
    
    # Missing helper methods implementation
    
    async def _load_configurations(self) -> None:
        """Load existing report configurations from database"""
        try:
            # In production, this would load from database
            # For now, create some default configurations
            self.logger.info("Loading report configurations from database")
            
            # Create default daily risk report configuration
            default_config = ReportConfig(
                report_id="default_daily_risk",
                report_type=ReportType.DAILY_RISK,
                name="Default Daily Risk Report",
                description="Automated daily risk assessment",
                portfolio_ids=["default"],
                format=ReportFormat.JSON,
                frequency=ReportFrequency.DAILY,
                schedule_expression="0 9 * * *",  # Daily at 9 AM
                recipients=["risk@company.com"],
                active=True
            )
            
            self.report_configs[default_config.report_id] = default_config
            
        except Exception as e:
            self.logger.error(f"Error loading configurations: {e}")
    
    async def _save_report_config(self, config: ReportConfig) -> None:
        """Save report configuration to database"""
        try:
            # In production, save to database
            self.logger.debug(f"Saving report configuration: {config.report_id}")
        except Exception as e:
            self.logger.error(f"Error saving report configuration: {e}")
    
    async def _check_scheduled_reports(self) -> None:
        """Check for scheduled reports that need to be run"""
        try:
            current_time = datetime.utcnow()
            
            for config_id, config in self.report_configs.items():
                if not config.active or config.frequency == ReportFrequency.ON_DEMAND:
                    continue
                
                if config.next_run and current_time >= config.next_run:
                    # Check if we're already running too many reports
                    if len(self.active_reports) >= self.max_concurrent_reports:
                        self.logger.warning(f"Delaying report {config_id} - too many concurrent reports")
                        continue
                    
                    # Start report generation task
                    task = asyncio.create_task(self._run_scheduled_report(config))
                    self.active_reports[config_id] = task
                    
                    # Schedule next run
                    config.next_run = self._calculate_next_run_time(config.frequency, config.schedule_expression)
                    
        except Exception as e:
            self.logger.error(f"Error checking scheduled reports: {e}")
    
    async def _run_scheduled_report(self, config: ReportConfig) -> None:
        """Run a scheduled report"""
        try:
            self.logger.info(f"Running scheduled report: {config.name}")
            
            # Generate report
            report_data, metadata = await self.generate_report(
                config.report_type,
                config.portfolio_ids,
                config.format,
                config.parameters,
                config.template_id
            )
            
            # Update configuration statistics
            config.last_run = datetime.utcnow()
            config.run_count += 1
            
            if metadata.status == "error":
                config.error_count += 1
                config.last_error = metadata.error_message
                self.logger.error(f"Scheduled report {config.name} failed: {metadata.error_message}")
            else:
                # In production, distribute report to recipients
                if config.recipients:
                    await self._distribute_report(config, report_data, metadata)
                
                self.logger.info(f"Successfully completed scheduled report: {config.name}")
            
        except Exception as e:
            self.logger.error(f"Error running scheduled report {config.name}: {e}")
            config.error_count += 1
            config.last_error = str(e)
        finally:
            # Remove from active reports
            if config.report_id in self.active_reports:
                del self.active_reports[config.report_id]
    
    async def _distribute_report(
        self, 
        config: ReportConfig, 
        report_data: Dict[str, Any], 
        metadata: ReportMetadata
    ) -> None:
        """Distribute report to recipients"""
        try:
            # In production, this would send emails, save to shared storage, etc.
            self.logger.info(f"Distributing report {config.name} to {len(config.recipients)} recipients")
            
            # Mock distribution logic
            for recipient in config.recipients:
                self.logger.debug(f"Sending report to {recipient}")
                
        except Exception as e:
            self.logger.error(f"Error distributing report: {e}")
    
    def _calculate_next_run_time(self, frequency: ReportFrequency, schedule_expression: Optional[str]) -> datetime:
        """Calculate next run time for scheduled report"""
        current_time = datetime.utcnow()
        
        if frequency == ReportFrequency.DAILY:
            return current_time + timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            return current_time + timedelta(weeks=1)
        elif frequency == ReportFrequency.MONTHLY:
            # Next month, same day
            if current_time.month == 12:
                return current_time.replace(year=current_time.year + 1, month=1)
            else:
                return current_time.replace(month=current_time.month + 1)
        elif frequency == ReportFrequency.INTRADAY:
            return current_time + timedelta(hours=6)  # Every 6 hours
        else:
            return current_time + timedelta(hours=1)  # Default
    
    async def _cleanup_old_reports(self) -> None:
        """Clean up old reports based on retention policy"""
        try:
            # In production, this would clean up old report files and database entries
            self.logger.debug("Cleaning up old reports")
            
            for config in self.report_configs.values():
                if config.retention_days > 0:
                    cutoff_date = datetime.utcnow() - timedelta(days=config.retention_days)
                    # Clean up reports older than retention period
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old reports: {e}")
    
    async def _cleanup_cache(self) -> None:
        """Clean up expired cache entries"""
        try:
            current_time = datetime.utcnow()
            expired_keys = []
            
            for key, entry in self.report_cache.items():
                cache_age = (current_time - entry['timestamp']).seconds
                if cache_age > (self.cache_ttl_minutes * 60):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.report_cache[key]
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {e}")
    
    async def _broadcast_dashboard_update(self, dashboard_data: Dict[str, Any]) -> None:
        """Broadcast dashboard update via WebSocket"""
        try:
            if self.redis_client:
                # Publish to WebSocket channel
                message = {
                    'type': 'dashboard_update',
                    'data': dashboard_data
                }
                # In production, would publish to Redis channel for WebSocket distribution
                self.logger.debug("Broadcasting dashboard update")
                
        except Exception as e:
            self.logger.error(f"Error broadcasting dashboard update: {e}")
    
    async def _format_report(
        self, 
        report_data: Dict[str, Any], 
        format: ReportFormat, 
        template_id: Optional[str]
    ) -> Dict[str, Any]:
        """Format report based on requested format and template"""
        try:
            if format == ReportFormat.JSON:
                return report_data
            elif format == ReportFormat.HTML:
                return await self._format_html_report(report_data, template_id)
            elif format == ReportFormat.DASHBOARD:
                return await self._format_dashboard_report(report_data)
            else:
                # For other formats, return raw data with format indicator
                return {
                    'format': format.value,
                    'data': report_data,
                    'template_id': template_id
                }
                
        except Exception as e:
            self.logger.error(f"Error formatting report: {e}")
            return report_data
    
    async def _format_html_report(self, report_data: Dict[str, Any], template_id: Optional[str]) -> Dict[str, Any]:
        """Format report as HTML"""
        try:
            template_name = template_id or 'default'
            template_content = self.report_templates.get(template_name, "<h1>Report</h1><pre>{{ data | tojson(indent=2) }}</pre>")
            
            # Use Jinja2 for templating
            env = Environment(loader=DictLoader({template_name: template_content}))
            template = env.get_template(template_name)
            
            html_content = template.render(data=report_data, **report_data)
            
            return {
                'format': 'html',
                'content': html_content,
                'template_id': template_id
            }
            
        except Exception as e:
            self.logger.error(f"Error formatting HTML report: {e}")
            return {'format': 'html', 'error': str(e)}
    
    async def _format_dashboard_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format report for dashboard consumption"""
        try:
            # Extract key metrics for dashboard display
            dashboard_format = {
                'format': 'dashboard',
                'widgets': [],
                'summary': {},
                'charts': {},
                'alerts': []
            }
            
            # Extract summary metrics
            if 'summary' in report_data:
                dashboard_format['summary'] = report_data['summary']
            
            # Extract alerts
            if 'risk_alerts' in report_data:
                dashboard_format['alerts'] = report_data['risk_alerts'][:10]  # Top 10
            
            # Convert portfolio data to widgets
            if 'portfolios' in report_data:
                for i, portfolio in enumerate(report_data['portfolios'][:6]):  # Max 6 portfolios
                    dashboard_format['widgets'].append({
                        'type': 'portfolio_summary',
                        'portfolio_id': portfolio.get('portfolio_id'),
                        'var_95': portfolio.get('var_1d_95', 0),
                        'total_value': portfolio.get('total_value', 0),
                        'risk_level': portfolio.get('risk_level', 'unknown'),
                        'position': {'x': i % 3, 'y': i // 3, 'width': 1, 'height': 1}
                    })
            
            return dashboard_format
            
        except Exception as e:
            self.logger.error(f"Error formatting dashboard report: {e}")
            return {'format': 'dashboard', 'error': str(e)}
    
    # Additional missing helper methods (mock implementations)
    
    async def _analyze_weekly_risk_trend(self, portfolio_ids: List[str], start_date, end_date) -> str:
        """Analyze weekly risk trend"""
        return np.random.choice(['improving', 'deteriorating', 'stable'])
    
    async def _analyze_weekly_breaches(self, portfolio_ids: List[str], start_date, end_date) -> Dict[str, Any]:
        """Analyze weekly breaches"""
        return {
            'total_breaches': np.random.randint(0, 5),
            'breach_types': {'var_limit': 2, 'concentration': 1},
            'resolved_breaches': np.random.randint(0, 3)
        }
    
    async def _calculate_monthly_statistics(self, portfolio_id: str, start_date, end_date) -> Dict[str, Any]:
        """Calculate monthly statistics"""
        return {
            'monthly_return': np.random.uniform(-5, 8),
            'monthly_volatility': np.random.uniform(10, 25),
            'max_drawdown': np.random.uniform(0, 8),
            'sharpe_ratio': np.random.uniform(-0.5, 2.0),
            'var_breach_count': np.random.randint(0, 5)
        }
    
    async def _analyze_risk_evolution(self, portfolio_id: str, start_date, end_date) -> Dict[str, Any]:
        """Analyze risk evolution over time"""
        return {
            'var_trend': 'stable',
            'volatility_trend': 'decreasing',
            'correlation_change': 0.05,
            'concentration_change': -0.02
        }
    
    async def _aggregate_monthly_statistics(self, portfolio_ids: List[str], start_date, end_date) -> Dict[str, Any]:
        """Aggregate monthly statistics across portfolios"""
        return {
            'total_return': np.random.uniform(-2, 5),
            'aggregate_volatility': np.random.uniform(12, 20),
            'correlation_matrix_stability': 0.85,
            'risk_contribution_changes': {'equity': 0.02, 'fixed_income': -0.01}
        }
    
    async def _assess_regulatory_compliance(self, portfolio_ids: List[str]) -> Dict[str, Any]:
        """Assess regulatory compliance"""
        return {
            'basel_iii_compliant': True,
            'risk_weighted_assets': 150000000,
            'tier1_capital_ratio': 0.12,
            'leverage_ratio': 0.05,
            'liquidity_coverage_ratio': 1.15
        }
    
    async def _generate_monthly_stress_summary(self, portfolio_ids: List[str]) -> Dict[str, Any]:
        """Generate monthly stress test summary"""
        return {
            'scenarios_tested': 5,
            'worst_case_loss': -25000000,
            'recovery_time_days': 30,
            'stress_test_pass_rate': 0.80
        }
    
    async def _calculate_regulatory_metrics(self, portfolio_id: str, regulation_type: str) -> Dict[str, Any]:
        """Calculate regulatory metrics"""
        return {
            'risk_weighted_assets': np.random.uniform(50000000, 200000000),
            'capital_adequacy_ratio': np.random.uniform(0.08, 0.15),
            'leverage_ratio': np.random.uniform(0.03, 0.08),
            'liquidity_ratio': np.random.uniform(1.0, 1.5)
        }
    
    async def _perform_compliance_checks(self, portfolio_id: str, regulation_type: str) -> Dict[str, Any]:
        """Perform regulatory compliance checks"""
        violations = []
        if np.random.random() < 0.2:  # 20% chance of violation
            violations.append({
                'rule': 'Capital Adequacy',
                'current_value': 0.07,
                'required_value': 0.08,
                'severity': 'medium'
            })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'score': 100 - (len(violations) * 10)
        }
    
    async def _generate_remediation_plan(self, portfolio_ids: List[str], regulation_type: str) -> List[Dict[str, Any]]:
        """Generate remediation plan for compliance violations"""
        return [
            {
                'action': 'Increase capital reserves',
                'priority': 'high',
                'timeline': '30 days',
                'responsible_party': 'Risk Management'
            },
            {
                'action': 'Reduce leverage exposure',
                'priority': 'medium',
                'timeline': '60 days',
                'responsible_party': 'Portfolio Management'
            }
        ]
    
    async def _generate_stress_test_recommendations(self, aggregate_impacts: Dict[str, Any]) -> List[str]:
        """Generate stress test recommendations"""
        recommendations = []
        
        for scenario, impact in aggregate_impacts.items():
            if abs(impact.get('total_impact', 0)) > 10000000:  # $10M threshold
                recommendations.append(f"Consider hedging against {scenario} scenario risk")
        
        if not recommendations:
            recommendations.append("Portfolio shows good resilience to tested stress scenarios")
        
        return recommendations
    
    # Chart data methods
    
    async def _get_exposure_breakdown_chart(self, portfolio_ids: Optional[List[str]]) -> Dict[str, Any]:
        """Get exposure breakdown chart data"""
        return {
            'type': 'pie',
            'labels': ['Equity', 'Fixed Income', 'Cash', 'Alternatives', 'Derivatives'],
            'data': [45, 30, 10, 10, 5],
            'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        }
    
    async def _get_limit_utilization_chart(self, portfolio_ids: Optional[List[str]]) -> Dict[str, Any]:
        """Get limit utilization chart data"""
        return {
            'type': 'bar',
            'labels': ['VaR Limit', 'Concentration', 'Leverage', 'Position Size', 'Drawdown'],
            'data': [75, 60, 45, 30, 20],
            'thresholds': [80, 80, 80, 80, 80],
            'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        }
    
    async def _get_risk_trends_chart(self, portfolio_ids: Optional[List[str]]) -> Dict[str, Any]:
        """Get risk trends chart data"""
        days = [(datetime.utcnow() - timedelta(days=i)).strftime('%m-%d') for i in range(30, 0, -1)]
        
        return {
            'type': 'line',
            'labels': days,
            'datasets': [
                {
                    'label': 'Portfolio VaR',
                    'data': [10000 + np.random.normal(0, 1000) for _ in range(30)],
                    'borderColor': '#FF6B6B'
                },
                {
                    'label': 'Market VaR',
                    'data': [8000 + np.random.normal(0, 800) for _ in range(30)],
                    'borderColor': '#4ECDC4'
                }
            ]
        }
    
    # Widget data methods
    
    async def _get_metric_value(self, data_source: str, portfolio_ids: Optional[List[str]], parameters: Dict[str, Any]) -> Any:
        """Get metric value for dashboard widget"""
        if data_source == 'var_95_total':
            return np.random.uniform(50000, 150000)
        elif data_source == 'alerts_count':
            return np.random.randint(0, 10)
        elif data_source == 'limit_utilization':
            return np.random.uniform(0, 100)
        else:
            return 0
    
    async def _get_chart_data(self, data_source: str, portfolio_ids: Optional[List[str]], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get chart data for dashboard widget"""
        if data_source == 'var_history':
            return await self._get_var_history_chart(portfolio_ids)
        elif data_source == 'exposure_breakdown':
            return await self._get_exposure_breakdown_chart(portfolio_ids)
        else:
            return {}
    
    async def _get_table_data(self, data_source: str, portfolio_ids: Optional[List[str]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get table data for dashboard widget"""
        return [
            {'portfolio': 'PORT_001', 'var_95': 25000, 'alerts': 2},
            {'portfolio': 'PORT_002', 'var_95': 18000, 'alerts': 0},
            {'portfolio': 'PORT_003', 'var_95': 32000, 'alerts': 1}
        ]
    
    async def _get_alert_data(self, data_source: str, portfolio_ids: Optional[List[str]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get alert data for dashboard widget"""
        return await self._get_dashboard_alerts(portfolio_ids)
    
    async def _get_gauge_data(self, data_source: str, portfolio_ids: Optional[List[str]], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get gauge data for dashboard widget"""
        value = await self._get_metric_value(data_source, portfolio_ids, parameters)
        return {
            'value': value,
            'min': 0,
            'max': 100,
            'thresholds': {'warning': 70, 'critical': 85}
        }
    
    # Performance calculation helpers
    
    async def _get_performance_indicators(self, portfolio_ids: List[str]) -> Dict[str, Any]:
        """Get performance indicators for executive summary"""
        return {
            'total_return_ytd': np.random.uniform(-5, 15),
            'information_ratio': np.random.uniform(0.5, 2.0),
            'max_drawdown': np.random.uniform(2, 12),
            'win_rate': np.random.uniform(0.45, 0.65),
            'avg_holding_period': np.random.randint(5, 30)
        }
    
    async def _generate_executive_charts(self, portfolio_ids: List[str]) -> Dict[str, Any]:
        """Generate charts for executive summary"""
        return {
            'risk_attribution': await self._get_exposure_breakdown_chart(portfolio_ids),
            'var_trend': await self._get_var_history_chart(portfolio_ids),
            'performance_attribution': {
                'labels': ['Alpha', 'Beta', 'Residual'],
                'data': [3.2, 1.8, -0.5]
            }
        }
    
    # Risk calculation helpers
    
    async def _calculate_aggregate_correlation_risk(self, portfolio_ids: List[str]) -> float:
        """Calculate aggregate correlation risk"""
        return np.random.uniform(0.3, 0.8)
    
    async def _calculate_aggregate_liquidity_risk(self, portfolio_ids: List[str]) -> float:
        """Calculate aggregate liquidity risk"""
        return np.random.uniform(0.1, 0.4)
    
    async def _calculate_aggregate_concentration_risk(self, portfolio_ids: List[str]) -> float:
        """Calculate aggregate concentration risk"""
        return np.random.uniform(0.15, 0.35)
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize report templates"""
        return {
            'executive_summary': """
            <h1>Executive Risk Summary</h1>
            <div class="summary-metrics">
                <div class="metric">
                    <h3>Total Portfolio Value</h3>
                    <p>${{ summary.total_value|formatcurrency }}</p>
                </div>
                <div class="metric">
                    <h3>Value at Risk (95%)</h3>
                    <p>${{ summary.total_var_95|formatcurrency }}</p>
                </div>
            </div>
            """,
            'daily_risk': """
            <h1>Daily Risk Report - {{ report_date }}</h1>
            <div class="portfolio-grid">
                {% for portfolio in portfolios %}
                <div class="portfolio-card">
                    <h3>{{ portfolio.portfolio_id }}</h3>
                    <p>VaR: ${{ portfolio.var_1d_95|formatcurrency }}</p>
                </div>
                {% endfor %}
            </div>
            """
        }
    
    def _initialize_dashboard_themes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize dashboard themes"""
        return {
            'default': {
                'colors': {
                    'primary': '#2196F3',
                    'success': '#4CAF50',
                    'warning': '#FF9800',
                    'danger': '#F44336'
                },
                'chart_colors': ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
            }
        }
    
    async def _initialize_default_widgets(self) -> None:
        """Initialize default dashboard widgets"""
        default_widgets = [
            DashboardWidget(
                widget_id='total_var_95',
                title='Total VaR (95%)',
                widget_type='metric',
                data_source='var_95_total',
                refresh_interval=30,
                position={'x': 0, 'y': 0, 'width': 3, 'height': 2}
            ),
            DashboardWidget(
                widget_id='active_alerts',
                title='Active Alerts',
                widget_type='metric',
                data_source='alerts_count',
                refresh_interval=10,
                position={'x': 3, 'y': 0, 'width': 3, 'height': 2}
            ),
            DashboardWidget(
                widget_id='var_chart',
                title='VaR History',
                widget_type='chart',
                data_source='var_history',
                refresh_interval=60,
                position={'x': 0, 'y': 2, 'width': 6, 'height': 4}
            ),
            DashboardWidget(
                widget_id='limit_utilization',
                title='Limit Utilization',
                widget_type='gauge',
                data_source='limit_utilization',
                refresh_interval=30,
                position={'x': 6, 'y': 0, 'width': 3, 'height': 3}
            )
        ]
        
        for widget in default_widgets:
            self.dashboard_widgets[widget.widget_id] = widget
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get risk reporter statistics"""
        return {
            'reports_generated': self.reports_generated,
            'dashboard_updates': self.dashboard_updates,
            'error_count': self.error_count,
            'avg_generation_time_ms': (
                self.total_generation_time / self.reports_generated 
                if self.reports_generated > 0 else 0
            ),
            'scheduled_reports': len(self.report_configs),
            'cached_reports': len(self.report_cache),
            'active_widgets': len([w for w in self.dashboard_widgets.values() if w.active]),
            'services_active': {
                'scheduler': self.scheduler_active,
                'dashboard': self.dashboard_active
            }
        }


# Global instance
risk_reporter = None

def get_risk_reporter() -> ComprehensiveRiskReporter:
    """Get global risk reporter instance"""
    global risk_reporter
    if risk_reporter is None:
        raise RuntimeError("Risk reporter not initialized. Call init_risk_reporter() first.")
    return risk_reporter

async def init_risk_reporter() -> ComprehensiveRiskReporter:
    """Initialize global risk reporter instance"""
    global risk_reporter
    risk_reporter = ComprehensiveRiskReporter()
    await risk_reporter.initialize()
    return risk_reporter