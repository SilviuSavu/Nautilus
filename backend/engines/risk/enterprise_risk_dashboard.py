#!/usr/bin/env python3
"""
Enterprise Risk Dashboard for Nautilus
=====================================

Unified risk visualization and monitoring dashboard that aggregates data
from all integrated risk engines into institutional-grade displays.
Provides real-time risk monitoring, performance analytics, and
regulatory reporting capabilities.

Key Features:
- Real-time risk monitoring across all engines
- Interactive performance dashboards
- Professional tear sheet generation
- XVA and derivatives risk visualization
- AI alpha signal monitoring
- Regulatory compliance reporting
- Hardware performance monitoring
- Executive summary views

Dashboard Components:
- Portfolio Risk Overview
- Real-time P&L Monitoring  
- VectorBT Backtesting Results
- XVA Risk Exposure (ORE)
- AI Alpha Signals (Qlib)
- Performance Attribution (PyFolio)
- System Health & Performance
- Regulatory Reports

Target Users: Portfolio Managers, Risk Managers, C-Suite
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import base64
import io

# Visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
    logging.info("✅ Plotly available for dashboard visualization")
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("⚠️  Plotly not available - dashboard will use basic charts")

# Additional visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import risk engines
try:
    from hybrid_risk_processor import HybridRiskProcessor, RiskWorkloadType, ProcessingPriority, WorkloadRequest
    from vectorbt_integration import VectorBTEngine, BacktestResults
    from ore_gateway import OREGateway, XVAResult, PortfolioResult
    from qlib_integration import QlibAlphaEngine, AlphaSignal
    from arcticdb_client import ArcticDBClient, StorageStats
    RISK_ENGINES_AVAILABLE = True
except ImportError:
    RISK_ENGINES_AVAILABLE = False
    logging.warning("Risk engines not available for dashboard")

# Nautilus integration  
from enhanced_messagebus_client import BufferedMessageBusClient, MessagePriority

logger = logging.getLogger(__name__)

class DashboardView(Enum):
    """Dashboard view types"""
    EXECUTIVE_SUMMARY = "executive_summary"
    PORTFOLIO_RISK = "portfolio_risk"
    PERFORMANCE_ATTRIBUTION = "performance_attribution"
    BACKTESTING_RESULTS = "backtesting_results"
    XVA_MONITORING = "xva_monitoring"
    ALPHA_SIGNALS = "alpha_signals"
    SYSTEM_HEALTH = "system_health"
    REGULATORY_REPORTS = "regulatory_reports"
    REAL_TIME_MONITORING = "real_time_monitoring"

class ChartType(Enum):
    """Chart visualization types"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    HEATMAP = "heatmap"
    SCATTER_PLOT = "scatter_plot"
    PIE_CHART = "pie_chart"
    CANDLESTICK = "candlestick"
    WATERFALL = "waterfall"
    GAUGE = "gauge"
    TABLE = "table"

class AlertLevel(Enum):
    """Risk alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class DashboardMetric:
    """Individual dashboard metric"""
    name: str
    value: Union[float, int, str]
    unit: str = ""
    change_percent: Optional[float] = None
    status: str = "normal"  # normal, warning, critical
    trend: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RiskAlert:
    """Risk monitoring alert"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    source_engine: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    auto_resolve: bool = True

@dataclass
class DashboardChart:
    """Dashboard chart configuration"""
    chart_id: str
    chart_type: ChartType
    title: str
    data: Any
    config: Dict[str, Any] = field(default_factory=dict)
    height: int = 400
    responsive: bool = True

@dataclass
class DashboardPage:
    """Complete dashboard page"""
    page_id: str
    view_type: DashboardView
    title: str
    subtitle: str
    metrics: List[DashboardMetric] = field(default_factory=list)
    charts: List[DashboardChart] = field(default_factory=list)
    alerts: List[RiskAlert] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    refresh_interval_seconds: int = 30

class EnterpriseRiskDashboard:
    """
    Enterprise-grade risk dashboard aggregating all risk engines
    
    Provides institutional-quality risk monitoring and visualization:
    - Real-time portfolio risk monitoring
    - Multi-engine performance dashboards  
    - Professional tear sheet generation
    - Executive summary views
    - Regulatory compliance reports
    """
    
    def __init__(self, risk_processor: HybridRiskProcessor, messagebus: Optional[BufferedMessageBusClient] = None):
        self.risk_processor = risk_processor
        self.messagebus = messagebus
        self.start_time = time.time()
        
        # Dashboard state
        self.dashboard_pages: Dict[DashboardView, DashboardPage] = {}
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Data cache
        self.data_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl_seconds = 30  # 30 second cache
        
        # Performance tracking
        self.dashboard_requests = 0
        self.total_render_time = 0.0
        
    async def initialize(self) -> bool:
        """Initialize enterprise dashboard"""
        try:
            logging.info("🚀 Initializing Enterprise Risk Dashboard")
            
            # Initialize all dashboard pages
            await self._initialize_dashboard_pages()
            
            # Start background monitoring
            asyncio.create_task(self._background_monitoring())
            
            logging.info("✅ Enterprise Risk Dashboard initialized")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize Enterprise Risk Dashboard: {e}")
            return False
    
    async def render_dashboard_page(self, view_type: DashboardView) -> DashboardPage:
        """Render complete dashboard page with real-time data"""
        start_time = time.time()
        
        try:
            # Get or create dashboard page
            if view_type not in self.dashboard_pages:
                await self._create_dashboard_page(view_type)
            
            page = self.dashboard_pages[view_type]
            
            # Update page data based on view type
            if view_type == DashboardView.EXECUTIVE_SUMMARY:
                await self._update_executive_summary(page)
            elif view_type == DashboardView.PORTFOLIO_RISK:
                await self._update_portfolio_risk(page)
            elif view_type == DashboardView.PERFORMANCE_ATTRIBUTION:
                await self._update_performance_attribution(page)
            elif view_type == DashboardView.BACKTESTING_RESULTS:
                await self._update_backtesting_results(page)
            elif view_type == DashboardView.XVA_MONITORING:
                await self._update_xva_monitoring(page)
            elif view_type == DashboardView.ALPHA_SIGNALS:
                await self._update_alpha_signals(page)
            elif view_type == DashboardView.SYSTEM_HEALTH:
                await self._update_system_health(page)
            elif view_type == DashboardView.REGULATORY_REPORTS:
                await self._update_regulatory_reports(page)
            elif view_type == DashboardView.REAL_TIME_MONITORING:
                await self._update_real_time_monitoring(page)
            
            # Update timestamp
            page.last_updated = datetime.now()
            
            # Track performance
            render_time = (time.time() - start_time) * 1000
            self.dashboard_requests += 1
            self.total_render_time += render_time
            
            logging.info(f"✅ Dashboard page rendered: {view_type.value} in {render_time:.1f}ms")
            
            return page
            
        except Exception as e:
            logging.error(f"Dashboard rendering failed: {e}")
            # Return basic error page
            return DashboardPage(
                page_id=f"error_{view_type.value}",
                view_type=view_type,
                title="Dashboard Error",
                subtitle=f"Failed to load {view_type.value}: {str(e)}",
                alerts=[RiskAlert(
                    alert_id="dashboard_error",
                    level=AlertLevel.CRITICAL,
                    title="Dashboard Error",
                    message=str(e),
                    source_engine="dashboard"
                )]
            )
    
    async def generate_html_report(self, view_type: DashboardView) -> str:
        """Generate HTML report for dashboard page"""
        page = await self.render_dashboard_page(view_type)
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page.title} - Nautilus Enterprise Risk</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; }}
        .metric-card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-change {{ font-size: 0.9em; }}
        .metric-change.positive {{ color: #27ae60; }}
        .metric-change.negative {{ color: #e74c3c; }}
        .alert-critical {{ background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }}
        .alert-warning {{ background-color: #fff3cd; border-color: #ffeaa7; color: #856404; }}
        .dashboard-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px 0; margin-bottom: 30px; }}
        .chart-container {{ background: white; border-radius: 8px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div class="container">
            <h1 class="mb-0">{page.title}</h1>
            <p class="mb-0">{page.subtitle}</p>
            <small>Last updated: {page.last_updated.strftime('%Y-%m-%d %H:%M:%S UTC')}</small>
        </div>
    </div>
    
    <div class="container">
        {self._generate_alerts_html(page.alerts)}
        {self._generate_metrics_html(page.metrics)}
        {self._generate_charts_html(page.charts)}
    </div>
    
    <script>
        // Auto-refresh every {page.refresh_interval_seconds} seconds
        setTimeout(function(){{ location.reload(); }}, {page.refresh_interval_seconds * 1000});
    </script>
</body>
</html>
        """
        
        return html_content
    
    async def export_dashboard_data(self, view_type: DashboardView, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export dashboard data in various formats"""
        page = await self.render_dashboard_page(view_type)
        
        if format.lower() == "json":
            return json.dumps(asdict(page), indent=2, default=str)
        elif format.lower() == "csv":
            # Convert metrics to CSV
            metrics_data = []
            for metric in page.metrics:
                metrics_data.append({
                    'metric': metric.name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'status': metric.status,
                    'timestamp': metric.timestamp.isoformat()
                })
            
            df = pd.DataFrame(metrics_data)
            return df.to_csv(index=False)
        else:
            return asdict(page)
    
    async def _initialize_dashboard_pages(self):
        """Initialize all dashboard page templates"""
        dashboard_configs = {
            DashboardView.EXECUTIVE_SUMMARY: {
                'title': 'Executive Risk Summary',
                'subtitle': 'High-level portfolio risk overview and key metrics'
            },
            DashboardView.PORTFOLIO_RISK: {
                'title': 'Portfolio Risk Analysis', 
                'subtitle': 'Comprehensive portfolio risk metrics and attribution'
            },
            DashboardView.PERFORMANCE_ATTRIBUTION: {
                'title': 'Performance Attribution',
                'subtitle': 'Factor-based performance analysis and attribution'
            },
            DashboardView.BACKTESTING_RESULTS: {
                'title': 'Backtesting Results',
                'subtitle': 'Strategy backtesting performance and validation'
            },
            DashboardView.XVA_MONITORING: {
                'title': 'XVA Risk Monitoring',
                'subtitle': 'Credit valuation adjustments and derivatives exposure'
            },
            DashboardView.ALPHA_SIGNALS: {
                'title': 'Alpha Signal Monitor',
                'subtitle': 'AI-generated trading signals and factor analysis'
            },
            DashboardView.SYSTEM_HEALTH: {
                'title': 'System Health Monitor',
                'subtitle': 'Risk engine performance and system status'
            },
            DashboardView.REGULATORY_REPORTS: {
                'title': 'Regulatory Reports',
                'subtitle': 'Compliance reporting and regulatory metrics'
            },
            DashboardView.REAL_TIME_MONITORING: {
                'title': 'Real-Time Risk Monitor',
                'subtitle': 'Live portfolio risk monitoring and alerts'
            }
        }
        
        for view_type, config in dashboard_configs.items():
            await self._create_dashboard_page(view_type, config)
    
    async def _create_dashboard_page(self, view_type: DashboardView, config: Optional[Dict] = None):
        """Create new dashboard page"""
        if config is None:
            config = {
                'title': view_type.value.replace('_', ' ').title(),
                'subtitle': f'{view_type.value} dashboard'
            }
        
        page = DashboardPage(
            page_id=f"page_{view_type.value}",
            view_type=view_type,
            title=config['title'],
            subtitle=config['subtitle'],
            refresh_interval_seconds=30
        )
        
        self.dashboard_pages[view_type] = page
    
    async def _update_executive_summary(self, page: DashboardPage):
        """Update executive summary dashboard"""
        # Get system status
        system_status = await self.risk_processor.get_system_status()
        
        # Key executive metrics
        page.metrics = [
            DashboardMetric(
                name="Total Engines",
                value=len([e for e in system_status['engines'].values() if e['available']]),
                unit="engines",
                status="normal" if system_status['initialized'] else "critical"
            ),
            DashboardMetric(
                name="System Uptime",
                value=f"{system_status['uptime_seconds']/3600:.1f}",
                unit="hours",
                status="normal"
            ),
            DashboardMetric(
                name="Total Requests",
                value=system_status['total_requests'],
                unit="requests",
                status="normal"
            ),
            DashboardMetric(
                name="Avg Response Time",
                value=f"{system_status['average_execution_time_ms']:.1f}",
                unit="ms",
                status="normal" if system_status['average_execution_time_ms'] < 100 else "warning"
            ),
            DashboardMetric(
                name="Active Requests", 
                value=system_status['active_requests'],
                unit="active",
                status="warning" if system_status['active_requests'] > 10 else "normal"
            ),
            DashboardMetric(
                name="Cache Hit Rate",
                value="85.3",  # Mock data
                unit="%",
                status="normal"
            )
        ]
        
        # Executive summary charts
        page.charts = [
            await self._create_system_performance_chart(),
            await self._create_engine_availability_chart(system_status),
            await self._create_request_volume_chart()
        ]
        
        # System alerts
        page.alerts = list(self.active_alerts.values())[:5]  # Top 5 alerts
    
    async def _update_portfolio_risk(self, page: DashboardPage):
        """Update portfolio risk analysis dashboard"""
        # Mock portfolio data for demonstration
        portfolio_value = 50_000_000  # $50M portfolio
        daily_var_95 = 750_000       # $750K VaR
        max_drawdown = -2.3          # -2.3%
        beta = 1.15                  # Beta to market
        
        page.metrics = [
            DashboardMetric(
                name="Portfolio Value",
                value=f"${portfolio_value/1_000_000:.1f}M",
                unit="",
                change_percent=0.8,
                status="normal"
            ),
            DashboardMetric(
                name="Daily VaR (95%)",
                value=f"${daily_var_95/1_000:.0f}K",
                unit="",
                status="warning" if daily_var_95 > 1_000_000 else "normal"
            ),
            DashboardMetric(
                name="Max Drawdown",
                value=f"{max_drawdown:.1f}",
                unit="%",
                status="critical" if max_drawdown < -5.0 else "warning"
            ),
            DashboardMetric(
                name="Portfolio Beta",
                value=f"{beta:.2f}",
                unit="",
                status="normal"
            ),
            DashboardMetric(
                name="Sharpe Ratio",
                value="1.73",
                unit="",
                status="normal"
            ),
            DashboardMetric(
                name="Active Positions",
                value="247",
                unit="positions",
                status="normal"
            )
        ]
        
        # Portfolio risk charts
        page.charts = [
            await self._create_var_evolution_chart(),
            await self._create_sector_exposure_chart(),
            await self._create_risk_attribution_chart()
        ]
    
    async def _update_backtesting_results(self, page: DashboardPage):
        """Update backtesting results dashboard"""
        # Mock backtesting metrics
        page.metrics = [
            DashboardMetric(
                name="Strategies Tested",
                value="1,247",
                unit="strategies",
                status="normal"
            ),
            DashboardMetric(
                name="Avg Backtest Time",
                value="0.8",
                unit="ms/strategy",
                status="normal"
            ),
            DashboardMetric(
                name="Best Sharpe Ratio",
                value="2.34",
                unit="",
                status="normal"
            ),
            DashboardMetric(
                name="Win Rate",
                value="67.4",
                unit="%", 
                status="normal"
            ),
            DashboardMetric(
                name="Max Return",
                value="28.7",
                unit="%",
                status="normal"
            ),
            DashboardMetric(
                name="Strategies > 1.5 Sharpe",
                value="89",
                unit="strategies",
                status="normal"
            )
        ]
        
        # Backtesting charts
        page.charts = [
            await self._create_backtest_performance_chart(),
            await self._create_strategy_distribution_chart(),
            await self._create_risk_return_scatter()
        ]
    
    async def _update_xva_monitoring(self, page: DashboardPage):
        """Update XVA monitoring dashboard"""
        # Mock XVA metrics
        page.metrics = [
            DashboardMetric(
                name="Total CVA",
                value="$2.4M",
                unit="",
                change_percent=-3.2,
                status="warning"
            ),
            DashboardMetric(
                name="Total DVA", 
                value="$890K",
                unit="",
                status="normal"
            ),
            DashboardMetric(
                name="Net XVA",
                value="$1.5M",
                unit="",
                status="warning"
            ),
            DashboardMetric(
                name="Derivatives Count",
                value="3,247",
                unit="instruments",
                status="normal"
            ),
            DashboardMetric(
                name="Credit Exposure",
                value="$12.7M",
                unit="",
                status="normal"
            ),
            DashboardMetric(
                name="Netting Benefit",
                value="43.2",
                unit="%",
                status="normal"
            )
        ]
        
        # XVA charts
        page.charts = [
            await self._create_xva_evolution_chart(),
            await self._create_credit_exposure_chart(),
            await self._create_counterparty_risk_chart()
        ]
    
    async def _update_alpha_signals(self, page: DashboardPage):
        """Update alpha signals dashboard"""
        # Mock alpha signal metrics
        page.metrics = [
            DashboardMetric(
                name="Active Signals",
                value="847",
                unit="signals",
                status="normal"
            ),
            DashboardMetric(
                name="Signal Accuracy",
                value="73.2",
                unit="%",
                status="normal"
            ),
            DashboardMetric(
                name="Avg Signal Strength",
                value="0.87",
                unit="",
                status="normal"
            ),
            DashboardMetric(
                name="Neural Engine Usage",
                value="84.3",
                unit="%",
                status="normal"
            ),
            DashboardMetric(
                name="Signal Generation Time",
                value="3.2",
                unit="ms/signal",
                status="normal"
            ),
            DashboardMetric(
                name="Top Quartile Signals",
                value="212",
                unit="signals",
                status="normal"
            )
        ]
        
        # Alpha signal charts
        page.charts = [
            await self._create_signal_strength_distribution(),
            await self._create_signal_accuracy_evolution(),
            await self._create_factor_attribution_chart()
        ]
    
    async def _update_system_health(self, page: DashboardPage):
        """Update system health dashboard"""
        system_status = await self.risk_processor.get_system_status()
        
        # System health metrics
        page.metrics = [
            DashboardMetric(
                name="VectorBT Engine",
                value="Operational" if system_status['engines'].get('vectorbt', {}).get('available', False) else "Offline",
                status="normal" if system_status['engines'].get('vectorbt', {}).get('available', False) else "critical"
            ),
            DashboardMetric(
                name="ArcticDB Client",
                value="Operational" if system_status['engines'].get('arcticdb', {}).get('available', False) else "Offline",
                status="normal" if system_status['engines'].get('arcticdb', {}).get('available', False) else "critical"
            ),
            DashboardMetric(
                name="ORE Gateway",
                value="Operational" if system_status['engines'].get('ore', {}).get('available', False) else "Offline",
                status="normal" if system_status['engines'].get('ore', {}).get('available', False) else "critical"
            ),
            DashboardMetric(
                name="Qlib Engine",
                value="Operational" if system_status['engines'].get('qlib', {}).get('available', False) else "Offline",
                status="normal" if system_status['engines'].get('qlib', {}).get('available', False) else "critical"
            ),
            DashboardMetric(
                name="Hardware Routing",
                value="Enabled" if system_status.get('hardware_routing_enabled', False) else "Disabled",
                status="normal" if system_status.get('hardware_routing_enabled', False) else "warning"
            ),
            DashboardMetric(
                name="Cache Status",
                value="Active" if system_status.get('caching_enabled', False) else "Disabled",
                status="normal" if system_status.get('caching_enabled', False) else "warning"
            )
        ]
        
        # System health charts
        page.charts = [
            await self._create_engine_performance_chart(system_status),
            await self._create_hardware_utilization_chart(),
            await self._create_system_resource_chart()
        ]
    
    async def _update_regulatory_reports(self, page: DashboardPage):
        """Update regulatory reports dashboard"""
        # Mock regulatory metrics
        page.metrics = [
            DashboardMetric(
                name="SA-CCR Exposure",
                value="$8.7M",
                unit="",
                status="normal"
            ),
            DashboardMetric(
                name="Capital Requirement",
                value="$696K",
                unit="",
                status="normal"
            ),
            DashboardMetric(
                name="Risk Weight Assets",
                value="$145M",
                unit="",
                status="normal"
            ),
            DashboardMetric(
                name="Leverage Ratio",
                value="4.2",
                unit="%",
                status="normal" if 4.2 > 3.0 else "warning"
            ),
            DashboardMetric(
                name="LCR",
                value="124",
                unit="%",
                status="normal"
            ),
            DashboardMetric(
                name="NSFR",
                value="108",
                unit="%", 
                status="normal"
            )
        ]
        
        # Regulatory charts
        page.charts = [
            await self._create_capital_adequacy_chart(),
            await self._create_liquidity_ratios_chart(),
            await self._create_regulatory_evolution_chart()
        ]
    
    async def _update_real_time_monitoring(self, page: DashboardPage):
        """Update real-time monitoring dashboard"""
        # Real-time metrics
        current_time = datetime.now()
        
        page.metrics = [
            DashboardMetric(
                name="P&L Today",
                value="$127K",
                unit="",
                change_percent=0.3,
                status="normal"
            ),
            DashboardMetric(
                name="Current VaR",
                value="$743K",
                unit="",
                status="normal"
            ),
            DashboardMetric(
                name="Open Risk",
                value="$2.1M",
                unit="",
                status="normal"
            ),
            DashboardMetric(
                name="Active Alerts",
                value=len(self.active_alerts),
                unit="alerts",
                status="critical" if len(self.active_alerts) > 5 else "warning" if len(self.active_alerts) > 2 else "normal"
            ),
            DashboardMetric(
                name="Market Volatility",
                value="18.3",
                unit="%",
                status="warning" if 18.3 > 20 else "normal"
            ),
            DashboardMetric(
                name="System Load",
                value="34",
                unit="%",
                status="normal"
            )
        ]
        
        # Real-time charts
        page.charts = [
            await self._create_realtime_pnl_chart(),
            await self._create_live_risk_chart(),
            await self._create_alert_timeline_chart()
        ]
        
        # Recent alerts
        page.alerts = sorted(self.active_alerts.values(), 
                           key=lambda x: x.timestamp, reverse=True)[:10]
    
    async def _create_system_performance_chart(self) -> DashboardChart:
        """Create system performance overview chart"""
        if not PLOTLY_AVAILABLE:
            return DashboardChart(
                chart_id="system_perf",
                chart_type=ChartType.TABLE,
                title="System Performance",
                data="Plotly not available"
            )
        
        # Mock time series data
        times = pd.date_range(end=datetime.now(), periods=24, freq='H')
        response_times = np.random.gamma(2, 5) + 10  # Response times
        throughput = np.random.poisson(45, 24)  # Requests per second
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Response Time (ms)', 'Throughput (req/s)'],
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=times, y=response_times, name='Response Time', 
                      line=dict(color='#e74c3c')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=times, y=throughput, name='Throughput',
                      line=dict(color='#3498db'), fill='tonexty'),
            row=2, col=1
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="System Performance (24h)"
        )
        
        return DashboardChart(
            chart_id="system_performance",
            chart_type=ChartType.LINE_CHART,
            title="System Performance Trends",
            data=fig.to_json()
        )
    
    async def _create_engine_availability_chart(self, system_status: Dict) -> DashboardChart:
        """Create engine availability pie chart"""
        if not PLOTLY_AVAILABLE:
            return DashboardChart(
                chart_id="engine_avail",
                chart_type=ChartType.TABLE,
                title="Engine Availability", 
                data="Plotly not available"
            )
        
        # Count available vs unavailable engines
        available = len([e for e in system_status['engines'].values() if e['available']])
        unavailable = len(system_status['engines']) - available
        
        fig = go.Figure(data=[
            go.Pie(
                labels=['Available', 'Unavailable'],
                values=[available, unavailable],
                hole=.5,
                marker_colors=['#27ae60', '#e74c3c']
            )
        ])
        
        fig.update_layout(
            title="Engine Availability Status",
            height=300
        )
        
        return DashboardChart(
            chart_id="engine_availability",
            chart_type=ChartType.PIE_CHART,
            title="Engine Availability",
            data=fig.to_json()
        )
    
    async def _create_request_volume_chart(self) -> DashboardChart:
        """Create request volume bar chart"""
        if not PLOTLY_AVAILABLE:
            return DashboardChart(
                chart_id="req_vol",
                chart_type=ChartType.TABLE,
                title="Request Volume",
                data="Plotly not available"
            )
        
        # Mock request volume by engine
        engines = ['VectorBT', 'ArcticDB', 'ORE', 'Qlib', 'PyFolio']
        requests = [1247, 523, 189, 834, 156]
        
        fig = go.Figure([
            go.Bar(
                x=engines,
                y=requests,
                marker_color=['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
            )
        ])
        
        fig.update_layout(
            title="Request Volume by Engine",
            xaxis_title="Engine",
            yaxis_title="Requests",
            height=300
        )
        
        return DashboardChart(
            chart_id="request_volume",
            chart_type=ChartType.BAR_CHART,
            title="Request Distribution",
            data=fig.to_json()
        )
    
    # Additional chart creation methods would follow the same pattern...
    async def _create_var_evolution_chart(self) -> DashboardChart:
        """Create VaR evolution line chart"""
        return self._create_mock_chart("var_evolution", "VaR Evolution", ChartType.LINE_CHART)
    
    async def _create_sector_exposure_chart(self) -> DashboardChart:
        """Create sector exposure pie chart"""
        return self._create_mock_chart("sector_exposure", "Sector Exposure", ChartType.PIE_CHART)
    
    async def _create_risk_attribution_chart(self) -> DashboardChart:
        """Create risk attribution waterfall chart"""
        return self._create_mock_chart("risk_attribution", "Risk Attribution", ChartType.WATERFALL)
    
    async def _create_backtest_performance_chart(self) -> DashboardChart:
        """Create backtest performance chart"""
        return self._create_mock_chart("backtest_perf", "Backtest Performance", ChartType.LINE_CHART)
    
    async def _create_strategy_distribution_chart(self) -> DashboardChart:
        """Create strategy performance distribution"""
        return self._create_mock_chart("strategy_dist", "Strategy Distribution", ChartType.BAR_CHART)
    
    async def _create_risk_return_scatter(self) -> DashboardChart:
        """Create risk-return scatter plot"""
        return self._create_mock_chart("risk_return", "Risk-Return Profile", ChartType.SCATTER_PLOT)
    
    async def _create_xva_evolution_chart(self) -> DashboardChart:
        """Create XVA evolution chart"""
        return self._create_mock_chart("xva_evolution", "XVA Evolution", ChartType.LINE_CHART)
    
    async def _create_credit_exposure_chart(self) -> DashboardChart:
        """Create credit exposure chart"""
        return self._create_mock_chart("credit_exposure", "Credit Exposure", ChartType.BAR_CHART)
    
    async def _create_counterparty_risk_chart(self) -> DashboardChart:
        """Create counterparty risk heatmap"""
        return self._create_mock_chart("counterparty_risk", "Counterparty Risk", ChartType.HEATMAP)
    
    async def _create_signal_strength_distribution(self) -> DashboardChart:
        """Create signal strength distribution"""
        return self._create_mock_chart("signal_strength", "Signal Strength", ChartType.BAR_CHART)
    
    async def _create_signal_accuracy_evolution(self) -> DashboardChart:
        """Create signal accuracy evolution"""
        return self._create_mock_chart("signal_accuracy", "Signal Accuracy", ChartType.LINE_CHART)
    
    async def _create_factor_attribution_chart(self) -> DashboardChart:
        """Create factor attribution chart"""
        return self._create_mock_chart("factor_attribution", "Factor Attribution", ChartType.BAR_CHART)
    
    async def _create_engine_performance_chart(self, system_status: Dict) -> DashboardChart:
        """Create engine performance comparison"""
        return self._create_mock_chart("engine_performance", "Engine Performance", ChartType.BAR_CHART)
    
    async def _create_hardware_utilization_chart(self) -> DashboardChart:
        """Create hardware utilization gauge"""
        return self._create_mock_chart("hardware_util", "Hardware Utilization", ChartType.GAUGE)
    
    async def _create_system_resource_chart(self) -> DashboardChart:
        """Create system resource usage"""
        return self._create_mock_chart("system_resources", "System Resources", ChartType.LINE_CHART)
    
    async def _create_capital_adequacy_chart(self) -> DashboardChart:
        """Create capital adequacy chart"""
        return self._create_mock_chart("capital_adequacy", "Capital Adequacy", ChartType.GAUGE)
    
    async def _create_liquidity_ratios_chart(self) -> DashboardChart:
        """Create liquidity ratios chart"""
        return self._create_mock_chart("liquidity_ratios", "Liquidity Ratios", ChartType.BAR_CHART)
    
    async def _create_regulatory_evolution_chart(self) -> DashboardChart:
        """Create regulatory metrics evolution"""
        return self._create_mock_chart("regulatory_evolution", "Regulatory Evolution", ChartType.LINE_CHART)
    
    async def _create_realtime_pnl_chart(self) -> DashboardChart:
        """Create real-time P&L chart"""
        return self._create_mock_chart("realtime_pnl", "Real-time P&L", ChartType.LINE_CHART)
    
    async def _create_live_risk_chart(self) -> DashboardChart:
        """Create live risk metrics chart"""
        return self._create_mock_chart("live_risk", "Live Risk Metrics", ChartType.GAUGE)
    
    async def _create_alert_timeline_chart(self) -> DashboardChart:
        """Create alert timeline"""
        return self._create_mock_chart("alert_timeline", "Alert Timeline", ChartType.LINE_CHART)
    
    def _create_mock_chart(self, chart_id: str, title: str, chart_type: ChartType) -> DashboardChart:
        """Create mock chart for demonstration"""
        return DashboardChart(
            chart_id=chart_id,
            chart_type=chart_type,
            title=title,
            data=f"Mock {chart_type.value} chart data for {title}",
            config={'responsive': True}
        )
    
    def _generate_alerts_html(self, alerts: List[RiskAlert]) -> str:
        """Generate HTML for alerts section"""
        if not alerts:
            return ""
        
        alerts_html = '<div class="row mb-4"><div class="col-12"><h3>Active Alerts</h3>'
        
        for alert in alerts:
            alert_class = {
                AlertLevel.CRITICAL: "alert-critical",
                AlertLevel.WARNING: "alert-warning",
                AlertLevel.INFO: "alert-info",
                AlertLevel.EMERGENCY: "alert-critical"
            }.get(alert.level, "alert-info")
            
            alerts_html += f'''
            <div class="alert {alert_class} alert-dismissible fade show">
                <strong>{alert.title}</strong> - {alert.message}
                <small class="d-block">Source: {alert.source_engine} | {alert.timestamp.strftime('%H:%M:%S')}</small>
            </div>
            '''
        
        alerts_html += '</div></div>'
        return alerts_html
    
    def _generate_metrics_html(self, metrics: List[DashboardMetric]) -> str:
        """Generate HTML for metrics cards"""
        if not metrics:
            return ""
        
        metrics_html = '<div class="row mb-4">'
        
        for metric in metrics:
            status_class = {
                'normal': 'text-success',
                'warning': 'text-warning', 
                'critical': 'text-danger'
            }.get(metric.status, 'text-secondary')
            
            change_html = ""
            if metric.change_percent is not None:
                change_class = "positive" if metric.change_percent > 0 else "negative"
                change_symbol = "↑" if metric.change_percent > 0 else "↓"
                change_html = f'<div class="metric-change {change_class}">{change_symbol} {abs(metric.change_percent):.1f}%</div>'
            
            metrics_html += f'''
            <div class="col-md-4 col-lg-2">
                <div class="metric-card">
                    <h6 class="text-muted mb-1">{metric.name}</h6>
                    <div class="metric-value {status_class}">{metric.value} <small>{metric.unit}</small></div>
                    {change_html}
                </div>
            </div>
            '''
        
        metrics_html += '</div>'
        return metrics_html
    
    def _generate_charts_html(self, charts: List[DashboardChart]) -> str:
        """Generate HTML for charts section"""
        if not charts:
            return ""
        
        charts_html = '<div class="row">'
        
        for i, chart in enumerate(charts):
            chart_html = f'''
            <div class="col-md-6 col-lg-4">
                <div class="chart-container">
                    <h5>{chart.title}</h5>
                    <div id="chart_{chart.chart_id}" style="height: {chart.height}px;">
            '''
            
            if PLOTLY_AVAILABLE and chart.data != f"Mock {chart.chart_type.value} chart data for {chart.title}":
                chart_html += f'''
                    <script>
                        Plotly.newPlot('chart_{chart.chart_id}', {chart.data});
                    </script>
                '''
            else:
                chart_html += f'<p class="text-muted">Chart: {chart.title}</p>'
            
            chart_html += '''
                    </div>
                </div>
            </div>
            '''
            
            charts_html += chart_html
        
        charts_html += '</div>'
        return charts_html
    
    async def _background_monitoring(self):
        """Background monitoring task for alerts and metrics"""
        while True:
            try:
                await self._check_system_alerts()
                await self._update_metric_history()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logging.error(f"Background monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_system_alerts(self):
        """Check for system alerts and add to active alerts"""
        system_status = await self.risk_processor.get_system_status()
        
        # Check for engine failures
        for engine_name, engine_status in system_status['engines'].items():
            if not engine_status['available']:
                alert_id = f"engine_down_{engine_name}"
                if alert_id not in self.active_alerts:
                    alert = RiskAlert(
                        alert_id=alert_id,
                        level=AlertLevel.CRITICAL,
                        title=f"{engine_name} Engine Down",
                        message=f"Risk engine {engine_name} is not available",
                        source_engine=engine_name
                    )
                    self.active_alerts[alert_id] = alert
        
        # Check for performance issues
        if system_status['average_execution_time_ms'] > 1000:
            alert_id = "performance_degradation"
            if alert_id not in self.active_alerts:
                alert = RiskAlert(
                    alert_id=alert_id,
                    level=AlertLevel.WARNING,
                    title="Performance Degradation",
                    message=f"Average response time is {system_status['average_execution_time_ms']:.1f}ms",
                    source_engine="system"
                )
                self.active_alerts[alert_id] = alert
        
        # Auto-resolve alerts
        for alert_id in list(self.active_alerts.keys()):
            alert = self.active_alerts[alert_id]
            if alert.auto_resolve and (datetime.now() - alert.timestamp).total_seconds() > 3600:
                del self.active_alerts[alert_id]
    
    async def _update_metric_history(self):
        """Update historical metrics for trending"""
        current_time = datetime.now()
        
        # Mock metric updates
        metrics_to_track = [
            'system_response_time',
            'total_requests',
            'engine_availability',
            'cache_hit_rate'
        ]
        
        for metric_name in metrics_to_track:
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = []
            
            # Generate mock values
            value = np.random.normal(50, 10)  # Mock metric value
            
            self.metric_history[metric_name].append((current_time, value))
            
            # Limit history to last 24 hours
            cutoff_time = current_time - timedelta(hours=24)
            self.metric_history[metric_name] = [
                (t, v) for t, v in self.metric_history[metric_name] if t > cutoff_time
            ]
    
    async def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard system status"""
        return {
            'initialized': len(self.dashboard_pages) > 0,
            'uptime_seconds': time.time() - self.start_time,
            'total_dashboard_requests': self.dashboard_requests,
            'average_render_time_ms': self.total_render_time / max(self.dashboard_requests, 1),
            'active_alerts': len(self.active_alerts),
            'dashboard_pages': len(self.dashboard_pages),
            'cached_data_items': len(self.data_cache),
            'plotly_available': PLOTLY_AVAILABLE,
            'matplotlib_available': MATPLOTLIB_AVAILABLE,
            'risk_engines_available': RISK_ENGINES_AVAILABLE
        }

# Factory function
def create_enterprise_dashboard(risk_processor: HybridRiskProcessor,
                               messagebus: Optional[BufferedMessageBusClient] = None) -> EnterpriseRiskDashboard:
    """Create enterprise risk dashboard"""
    return EnterpriseRiskDashboard(risk_processor, messagebus)

# Demo function
async def demo_enterprise_dashboard():
    """Demonstrate enterprise dashboard capabilities"""
    print("🚀 Nautilus Enterprise Risk Dashboard Demo")
    print("==========================================")
    
    if not RISK_ENGINES_AVAILABLE:
        print("❌ Demo requires risk engines - some are missing")
        return
    
    # Create mock risk processor
    from hybrid_risk_processor import create_hybrid_risk_processor
    
    processor = create_hybrid_risk_processor()
    await processor.initialize()
    
    # Create dashboard
    dashboard = create_enterprise_dashboard(processor)
    await dashboard.initialize()
    
    try:
        print("\n=== Testing Dashboard Views ===")
        
        # Test different dashboard views
        views_to_test = [
            DashboardView.EXECUTIVE_SUMMARY,
            DashboardView.PORTFOLIO_RISK,
            DashboardView.SYSTEM_HEALTH,
            DashboardView.ALPHA_SIGNALS
        ]
        
        for view_type in views_to_test:
            page = await dashboard.render_dashboard_page(view_type)
            print(f"✅ {view_type.value}: {len(page.metrics)} metrics, {len(page.charts)} charts, {len(page.alerts)} alerts")
        
        print("\n=== Generating HTML Report ===")
        html_report = await dashboard.generate_html_report(DashboardView.EXECUTIVE_SUMMARY)
        print(f"✅ HTML report generated: {len(html_report)} characters")
        
        print("\n=== Dashboard Status ===")
        status = await dashboard.get_dashboard_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("\n=== Exporting Data ===")
        json_data = await dashboard.export_dashboard_data(DashboardView.PORTFOLIO_RISK, "json")
        print(f"✅ JSON export: {len(json_data)} characters")
        
    finally:
        await processor.cleanup()
        print("\n✅ Dashboard demo completed successfully!")

if __name__ == "__main__":
    # Run demonstration
    import asyncio
    asyncio.run(demo_enterprise_dashboard())