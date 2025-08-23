"""
Risk Reporting and Dashboard System
Real-time risk dashboards, report generation, and regulatory compliance
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta, time
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass, asdict
import json
import pandas as pd
import numpy as np
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of risk reports"""
    DAILY_RISK = "daily_risk"
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_SUMMARY = "monthly_summary"
    REGULATORY = "regulatory"
    STRESS_TEST = "stress_test"
    LIMIT_BREACH = "limit_breach"
    VAR_BACKTEST = "var_backtest"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"
    CUSTOM = "custom"

class ReportFormat(Enum):
    """Report output formats"""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    HTML = "html"

@dataclass
class ReportConfig:
    """Report configuration"""
    report_type: ReportType
    name: str
    description: str
    portfolio_ids: List[str]
    schedule: Optional[str] = None  # Cron expression
    format: ReportFormat = ReportFormat.PDF
    recipients: List[str] = None
    parameters: Dict[str, Any] = None
    active: bool = True
    created_at: datetime = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None

@dataclass
class DashboardMetric:
    """Real-time dashboard metric"""
    metric_id: str
    name: str
    value: Union[str, float, int]
    unit: str
    change_1h: Optional[float] = None
    change_24h: Optional[float] = None
    status: str = "normal"  # normal, warning, critical
    threshold: Optional[float] = None
    timestamp: datetime = None

class RiskReporter:
    """
    Comprehensive risk reporting system with real-time dashboards,
    automated report generation, and regulatory compliance
    """
    
    def __init__(self, risk_monitor=None, limit_engine=None, breach_detector=None, risk_calculator=None):
        self.risk_monitor = risk_monitor
        self.limit_engine = limit_engine
        self.breach_detector = breach_detector
        self.risk_calculator = risk_calculator
        
        # Report configurations
        self.report_configs: Dict[str, ReportConfig] = {}
        
        # Dashboard state
        self.dashboard_metrics: Dict[str, DashboardMetric] = {}
        self.dashboard_subscribers: Dict[str, List] = {}  # WebSocket connections
        
        # Scheduled reports
        self.scheduler_active = False
        self._scheduler_task = None
        
        # Report cache
        self.report_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_minutes = 30
        
        # Dashboard update frequency
        self.dashboard_update_interval = 5.0  # seconds
        self._dashboard_task = None
    
    async def start_services(self):
        """Start reporting services"""
        try:
            logger.info("Starting risk reporting services")
            
            # Start dashboard updates
            self._dashboard_task = asyncio.create_task(self._dashboard_update_loop())
            
            # Start report scheduler
            self.scheduler_active = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            
            logger.info("Risk reporting services started")
            
        except Exception as e:
            logger.error(f"Error starting reporting services: {e}")
            raise
    
    async def stop_services(self):
        """Stop reporting services"""
        try:
            logger.info("Stopping risk reporting services")
            
            self.scheduler_active = False
            
            # Stop tasks
            for task in [self._dashboard_task, self._scheduler_task]:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            logger.info("Risk reporting services stopped")
            
        except Exception as e:
            logger.error(f"Error stopping reporting services: {e}")
            raise
    
    async def _dashboard_update_loop(self):
        """Real-time dashboard update loop"""
        try:
            while True:
                start_time = datetime.utcnow()
                
                # Update dashboard metrics
                await self._update_dashboard_metrics()
                
                # Broadcast updates to subscribers
                await self._broadcast_dashboard_updates()
                
                # Sleep for update interval
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                sleep_time = max(0, self.dashboard_update_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Dashboard update loop cancelled")
        except Exception as e:
            logger.error(f"Error in dashboard update loop: {e}")
    
    async def _scheduler_loop(self):
        """Report scheduler loop"""
        try:
            while self.scheduler_active:
                # Check for scheduled reports
                await self._check_scheduled_reports()
                
                # Sleep for 1 minute
                await asyncio.sleep(60)
                
        except asyncio.CancelledError:
            logger.info("Report scheduler loop cancelled")
        except Exception as e:
            logger.error(f"Error in report scheduler loop: {e}")
            self.scheduler_active = False
    
    async def _update_dashboard_metrics(self):
        """Update real-time dashboard metrics"""
        try:
            current_time = datetime.utcnow()
            
            # Portfolio risk metrics
            await self._update_portfolio_metrics()
            
            # System health metrics
            await self._update_system_health_metrics()
            
            # Alert metrics
            await self._update_alert_metrics()
            
            # Limit utilization metrics
            await self._update_limit_metrics()
            
        except Exception as e:
            logger.error(f"Error updating dashboard metrics: {e}")
    
    async def _update_portfolio_metrics(self):
        """Update portfolio-level risk metrics"""
        try:
            if not self.risk_monitor:
                return
            
            # Get monitored portfolios
            status = await self.risk_monitor.get_monitoring_status()
            portfolios = status.get('monitored_portfolios', [])
            
            total_var = Decimal('0')
            total_exposure = Decimal('0')
            max_concentration = 0.0
            avg_beta = 0.0
            
            portfolio_count = 0
            
            for portfolio_id in portfolios:
                snapshot = await self.risk_monitor.get_current_risk_metrics(portfolio_id)
                if snapshot:
                    total_var += snapshot.var_1d_95
                    total_exposure += snapshot.total_exposure
                    max_concentration = max(max_concentration, snapshot.max_position_concentration)
                    avg_beta += snapshot.portfolio_beta
                    portfolio_count += 1
            
            if portfolio_count > 0:
                avg_beta /= portfolio_count
            
            # Update metrics
            self.dashboard_metrics['total_var_95'] = DashboardMetric(
                metric_id='total_var_95',
                name='Total VaR (95%)',
                value=float(total_var),
                unit='USD',
                timestamp=datetime.utcnow()
            )
            
            self.dashboard_metrics['total_exposure'] = DashboardMetric(
                metric_id='total_exposure',
                name='Total Exposure',
                value=float(total_exposure),
                unit='USD',
                timestamp=datetime.utcnow()
            )
            
            self.dashboard_metrics['max_concentration'] = DashboardMetric(
                metric_id='max_concentration',
                name='Max Position Concentration',
                value=max_concentration * 100,
                unit='%',
                status='warning' if max_concentration > 0.25 else 'normal',
                threshold=25.0,
                timestamp=datetime.utcnow()
            )
            
            self.dashboard_metrics['avg_beta'] = DashboardMetric(
                metric_id='avg_beta',
                name='Average Portfolio Beta',
                value=avg_beta,
                unit='',
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    async def _update_system_health_metrics(self):
        """Update system health metrics"""
        try:
            # Risk monitoring status
            if self.risk_monitor:
                status = await self.risk_monitor.get_monitoring_status()
                self.dashboard_metrics['risk_monitoring_active'] = DashboardMetric(
                    metric_id='risk_monitoring_active',
                    name='Risk Monitoring',
                    value='Active' if status.get('monitoring_active') else 'Inactive',
                    unit='',
                    status='normal' if status.get('monitoring_active') else 'critical',
                    timestamp=datetime.utcnow()
                )
            
            # Limit engine status
            if self.limit_engine:
                status = await self.limit_engine.get_limit_status()
                active_limits = status.get('total_limits', 0)
                
                self.dashboard_metrics['active_limits'] = DashboardMetric(
                    metric_id='active_limits',
                    name='Active Risk Limits',
                    value=active_limits,
                    unit='limits',
                    timestamp=datetime.utcnow()
                )
            
        except Exception as e:
            logger.error(f"Error updating system health metrics: {e}")
    
    async def _update_alert_metrics(self):
        """Update alert metrics"""
        try:
            if not self.breach_detector:
                return
            
            # Get active alerts
            active_alerts = await self.breach_detector.get_active_alerts()
            
            # Count by severity
            severity_counts = {}
            for alert in active_alerts:
                severity = alert.get('severity', 'unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            self.dashboard_metrics['active_alerts'] = DashboardMetric(
                metric_id='active_alerts',
                name='Active Alerts',
                value=len(active_alerts),
                unit='alerts',
                status='critical' if len(active_alerts) > 0 else 'normal',
                timestamp=datetime.utcnow()
            )
            
            # Critical alerts
            critical_count = severity_counts.get('critical', 0) + severity_counts.get('emergency', 0)
            self.dashboard_metrics['critical_alerts'] = DashboardMetric(
                metric_id='critical_alerts',
                name='Critical Alerts',
                value=critical_count,
                unit='alerts',
                status='critical' if critical_count > 0 else 'normal',
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error updating alert metrics: {e}")
    
    async def _update_limit_metrics(self):
        """Update limit utilization metrics"""
        try:
            if not self.limit_engine:
                return
            
            status = await self.limit_engine.get_limit_status()
            limits = status.get('limits', [])
            
            # Calculate average utilization
            total_utilization = 0
            limit_count = 0
            max_utilization = 0
            
            for limit in limits:
                if limit.get('active') and limit.get('utilization_pct') is not None:
                    utilization = limit['utilization_pct']
                    total_utilization += utilization
                    max_utilization = max(max_utilization, utilization)
                    limit_count += 1
            
            avg_utilization = total_utilization / limit_count if limit_count > 0 else 0
            
            self.dashboard_metrics['avg_limit_utilization'] = DashboardMetric(
                metric_id='avg_limit_utilization',
                name='Avg Limit Utilization',
                value=avg_utilization,
                unit='%',
                status='warning' if avg_utilization > 80 else 'normal',
                threshold=80.0,
                timestamp=datetime.utcnow()
            )
            
            self.dashboard_metrics['max_limit_utilization'] = DashboardMetric(
                metric_id='max_limit_utilization',
                name='Max Limit Utilization',
                value=max_utilization,
                unit='%',
                status='critical' if max_utilization > 95 else 'warning' if max_utilization > 80 else 'normal',
                threshold=95.0,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error updating limit metrics: {e}")
    
    async def _broadcast_dashboard_updates(self):
        """Broadcast dashboard updates to subscribers"""
        try:
            # Convert metrics to dict for JSON serialization
            metrics_data = {}
            for metric_id, metric in self.dashboard_metrics.items():
                metrics_data[metric_id] = {
                    'name': metric.name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'status': metric.status,
                    'change_1h': metric.change_1h,
                    'change_24h': metric.change_24h,
                    'threshold': metric.threshold,
                    'timestamp': metric.timestamp.isoformat() if metric.timestamp else None
                }
            
            # Broadcast to all dashboard subscribers
            # Implementation would use WebSocket manager
            logger.debug(f"Broadcasting dashboard update with {len(metrics_data)} metrics")
            
        except Exception as e:
            logger.error(f"Error broadcasting dashboard updates: {e}")
    
    async def generate_report(self, report_type: ReportType, portfolio_ids: List[str], 
                            parameters: Optional[Dict[str, Any]] = None,
                            format: ReportFormat = ReportFormat.JSON) -> Dict[str, Any]:
        """Generate a risk report"""
        try:
            logger.info(f"Generating {report_type.value} report for portfolios: {portfolio_ids}")
            
            # Check cache first
            cache_key = f"{report_type.value}_{'-'.join(portfolio_ids)}_{hash(str(parameters))}"
            cached_report = self._get_cached_report(cache_key)
            if cached_report:
                return cached_report
            
            # Generate report based on type
            if report_type == ReportType.DAILY_RISK:
                report_data = await self._generate_daily_risk_report(portfolio_ids, parameters)
            elif report_type == ReportType.WEEKLY_SUMMARY:
                report_data = await self._generate_weekly_summary_report(portfolio_ids, parameters)
            elif report_type == ReportType.MONTHLY_SUMMARY:
                report_data = await self._generate_monthly_summary_report(portfolio_ids, parameters)
            elif report_type == ReportType.REGULATORY:
                report_data = await self._generate_regulatory_report(portfolio_ids, parameters)
            elif report_type == ReportType.STRESS_TEST:
                report_data = await self._generate_stress_test_report(portfolio_ids, parameters)
            elif report_type == ReportType.LIMIT_BREACH:
                report_data = await self._generate_limit_breach_report(portfolio_ids, parameters)
            elif report_type == ReportType.VAR_BACKTEST:
                report_data = await self._generate_var_backtest_report(portfolio_ids, parameters)
            elif report_type == ReportType.CORRELATION:
                report_data = await self._generate_correlation_report(portfolio_ids, parameters)
            elif report_type == ReportType.CONCENTRATION:
                report_data = await self._generate_concentration_report(portfolio_ids, parameters)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            # Format report
            formatted_report = await self._format_report(report_data, format)
            
            # Cache report
            self._cache_report(cache_key, formatted_report)
            
            logger.info(f"Successfully generated {report_type.value} report")
            return formatted_report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    async def _generate_daily_risk_report(self, portfolio_ids: List[str], parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate daily risk report"""
        try:
            report_date = parameters.get('date', datetime.utcnow().date()) if parameters else datetime.utcnow().date()
            
            report_data = {
                'report_type': 'Daily Risk Report',
                'date': report_date.isoformat(),
                'portfolios': [],
                'summary': {},
                'generated_at': datetime.utcnow().isoformat()
            }
            
            total_var = Decimal('0')
            total_exposure = Decimal('0')
            
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_portfolio_risk_data(portfolio_id)
                report_data['portfolios'].append(portfolio_data)
                
                total_var += Decimal(str(portfolio_data.get('var_1d_95', 0)))
                total_exposure += Decimal(str(portfolio_data.get('total_exposure', 0)))
            
            # Summary section
            report_data['summary'] = {
                'total_var_1d_95': str(total_var),
                'total_exposure': str(total_exposure),
                'portfolio_count': len(portfolio_ids),
                'risk_utilization': float(total_var / total_exposure * 100) if total_exposure > 0 else 0
            }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating daily risk report: {e}")
            raise
    
    async def _generate_weekly_summary_report(self, portfolio_ids: List[str], parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate weekly summary report"""
        try:
            end_date = parameters.get('end_date', datetime.utcnow().date()) if parameters else datetime.utcnow().date()
            start_date = end_date - timedelta(days=7)
            
            report_data = {
                'report_type': 'Weekly Risk Summary',
                'period': f"{start_date.isoformat()} to {end_date.isoformat()}",
                'portfolios': [],
                'weekly_trends': {},
                'breach_summary': {},
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Get breach statistics for the week
            if self.breach_detector:
                breach_stats = await self.breach_detector.get_breach_statistics(hours=7*24)
                report_data['breach_summary'] = breach_stats
            
            # Portfolio data
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_portfolio_risk_data(portfolio_id)
                
                # Add weekly trend analysis
                trend_data = await self._get_weekly_trends(portfolio_id, start_date, end_date)
                portfolio_data.update(trend_data)
                
                report_data['portfolios'].append(portfolio_data)
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating weekly summary report: {e}")
            raise
    
    async def _generate_monthly_summary_report(self, portfolio_ids: List[str], parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate monthly summary report"""
        # Similar to weekly but with monthly data
        return await self._generate_weekly_summary_report(portfolio_ids, parameters)
    
    async def _generate_regulatory_report(self, portfolio_ids: List[str], parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate regulatory compliance report"""
        try:
            report_data = {
                'report_type': 'Regulatory Compliance Report',
                'regulation': parameters.get('regulation', 'Basel III') if parameters else 'Basel III',
                'portfolios': [],
                'compliance_summary': {},
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Add regulatory-specific metrics
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_portfolio_risk_data(portfolio_id)
                
                # Add regulatory metrics
                regulatory_metrics = await self._calculate_regulatory_metrics(portfolio_id, parameters)
                portfolio_data['regulatory_metrics'] = regulatory_metrics
                
                report_data['portfolios'].append(portfolio_data)
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating regulatory report: {e}")
            raise
    
    async def _generate_stress_test_report(self, portfolio_ids: List[str], parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate stress test report"""
        try:
            scenario = parameters.get('scenario', 'Market Crash') if parameters else 'Market Crash'
            
            report_data = {
                'report_type': 'Stress Test Report',
                'scenario': scenario,
                'portfolios': [],
                'stress_results': {},
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Mock stress test results
            for portfolio_id in portfolio_ids:
                portfolio_data = await self._get_portfolio_risk_data(portfolio_id)
                
                # Add stress test results
                stress_results = {
                    'baseline_var': portfolio_data.get('var_1d_95', 0),
                    'stressed_var': float(Decimal(str(portfolio_data.get('var_1d_95', 0))) * Decimal('1.5')),
                    'pnl_impact': -50000,  # Mock value
                    'worst_position': 'AAPL',
                    'sector_impacts': {'Technology': -15.5, 'Healthcare': -8.2}
                }
                
                portfolio_data['stress_results'] = stress_results
                report_data['portfolios'].append(portfolio_data)
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating stress test report: {e}")
            raise
    
    async def _generate_limit_breach_report(self, portfolio_ids: List[str], parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate limit breach report"""
        try:
            hours = parameters.get('hours', 24) if parameters else 24
            
            report_data = {
                'report_type': 'Limit Breach Report',
                'time_period_hours': hours,
                'breaches': [],
                'summary': {},
                'generated_at': datetime.utcnow().isoformat()
            }
            
            if self.breach_detector:
                for portfolio_id in portfolio_ids:
                    alerts = await self.breach_detector.get_active_alerts(portfolio_id)
                    report_data['breaches'].extend(alerts)
                
                # Get statistics
                breach_stats = await self.breach_detector.get_breach_statistics(hours=hours)
                report_data['summary'] = breach_stats
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating limit breach report: {e}")
            raise
    
    async def _generate_var_backtest_report(self, portfolio_ids: List[str], parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate VaR backtest report"""
        # Mock VaR backtest implementation
        return {
            'report_type': 'VaR Backtest Report',
            'portfolios': portfolio_ids,
            'backtest_results': {
                'coverage_ratio_95': 0.94,
                'coverage_ratio_99': 0.99,
                'kupiec_test_p_value': 0.15,
                'exceptions_count': 12,
                'total_observations': 252
            },
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def _generate_correlation_report(self, portfolio_ids: List[str], parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate correlation analysis report"""
        # Mock correlation analysis
        return {
            'report_type': 'Correlation Analysis Report',
            'portfolios': portfolio_ids,
            'correlation_matrix': [[1.0, 0.7], [0.7, 1.0]],
            'high_correlations': [{'asset1': 'AAPL', 'asset2': 'MSFT', 'correlation': 0.85}],
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def _generate_concentration_report(self, portfolio_ids: List[str], parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate concentration risk report"""
        # Mock concentration analysis
        return {
            'report_type': 'Concentration Risk Report',
            'portfolios': portfolio_ids,
            'concentration_metrics': {
                'herfindahl_index': 0.15,
                'top_5_concentration': 0.65,
                'largest_position_pct': 0.25
            },
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def _get_portfolio_risk_data(self, portfolio_id: str) -> Dict[str, Any]:
        """Get comprehensive risk data for a portfolio"""
        try:
            portfolio_data = {
                'portfolio_id': portfolio_id,
                'var_1d_95': 0,
                'var_1d_99': 0,
                'total_exposure': 0,
                'net_exposure': 0,
                'max_concentration': 0,
                'portfolio_beta': 1.0,
                'portfolio_volatility': 0.15,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Get data from risk monitor if available
            if self.risk_monitor:
                snapshot = await self.risk_monitor.get_current_risk_metrics(portfolio_id)
                if snapshot:
                    portfolio_data.update({
                        'var_1d_95': str(snapshot.var_1d_95),
                        'var_1d_99': str(snapshot.var_1d_99),
                        'total_exposure': str(snapshot.total_exposure),
                        'net_exposure': str(snapshot.net_exposure),
                        'max_concentration': snapshot.max_position_concentration,
                        'portfolio_beta': snapshot.portfolio_beta,
                        'portfolio_volatility': snapshot.portfolio_volatility
                    })
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error getting portfolio risk data: {e}")
            return {'portfolio_id': portfolio_id, 'error': str(e)}
    
    async def _get_weekly_trends(self, portfolio_id: str, start_date, end_date) -> Dict[str, Any]:
        """Get weekly trend data for a portfolio"""
        # Mock trend analysis
        return {
            'var_trend': 'increasing',
            'var_change_pct': 5.2,
            'volatility_trend': 'stable',
            'exposure_change_pct': -2.1
        }
    
    async def _calculate_regulatory_metrics(self, portfolio_id: str, parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate regulatory-specific metrics"""
        # Mock regulatory calculations
        return {
            'risk_weighted_assets': 150000000,
            'tier1_capital_ratio': 0.12,
            'leverage_ratio': 0.05,
            'liquidity_coverage_ratio': 1.15
        }
    
    async def _format_report(self, report_data: Dict[str, Any], format: ReportFormat) -> Dict[str, Any]:
        """Format report based on requested format"""
        try:
            if format == ReportFormat.JSON:
                return {
                    'format': 'json',
                    'data': report_data,
                    'size_bytes': len(json.dumps(report_data))
                }
            elif format == ReportFormat.PDF:
                # Mock PDF generation
                return {
                    'format': 'pdf',
                    'data': base64.b64encode(b'mock-pdf-content').decode(),
                    'filename': f"risk_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
                }
            elif format == ReportFormat.EXCEL:
                # Mock Excel generation
                return {
                    'format': 'excel',
                    'data': base64.b64encode(b'mock-excel-content').decode(),
                    'filename': f"risk_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
                }
            elif format == ReportFormat.CSV:
                # Mock CSV generation
                return {
                    'format': 'csv',
                    'data': 'portfolio_id,var_1d_95,total_exposure\ntest,1000,50000',
                    'filename': f"risk_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
                }
            else:
                return {'format': 'json', 'data': report_data}
                
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            return {'format': 'json', 'data': report_data}
    
    def _get_cached_report(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached report if still valid"""
        try:
            if cache_key in self.report_cache:
                cached_report = self.report_cache[cache_key]
                cache_time = cached_report.get('cached_at')
                
                if cache_time:
                    cache_age_minutes = (datetime.utcnow() - cache_time).total_seconds() / 60
                    if cache_age_minutes < self.cache_ttl_minutes:
                        logger.debug(f"Using cached report: {cache_key}")
                        return cached_report['data']
                    else:
                        # Cache expired
                        del self.report_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking report cache: {e}")
            return None
    
    def _cache_report(self, cache_key: str, report_data: Dict[str, Any]):
        """Cache report data"""
        try:
            self.report_cache[cache_key] = {
                'data': report_data,
                'cached_at': datetime.utcnow()
            }
            
            # Cleanup old cache entries
            if len(self.report_cache) > 100:  # Keep max 100 cached reports
                oldest_key = min(self.report_cache.keys(), 
                               key=lambda k: self.report_cache[k]['cached_at'])
                del self.report_cache[oldest_key]
                
        except Exception as e:
            logger.error(f"Error caching report: {e}")
    
    async def _check_scheduled_reports(self):
        """Check and run scheduled reports"""
        try:
            current_time = datetime.utcnow()
            
            for config_id, config in self.report_configs.items():
                if not config.active or not config.schedule:
                    continue
                
                # Simple scheduling check (would use proper cron parser in production)
                if config.next_run and current_time >= config.next_run:
                    await self._run_scheduled_report(config)
                    
        except Exception as e:
            logger.error(f"Error checking scheduled reports: {e}")
    
    async def _run_scheduled_report(self, config: ReportConfig):
        """Run a scheduled report"""
        try:
            logger.info(f"Running scheduled report: {config.name}")
            
            # Generate report
            report = await self.generate_report(
                config.report_type,
                config.portfolio_ids,
                config.parameters,
                config.format
            )
            
            # Send to recipients (mock implementation)
            if config.recipients:
                logger.info(f"Sending report {config.name} to {len(config.recipients)} recipients")
            
            # Update last run time
            config.last_run = datetime.utcnow()
            config.next_run = self._calculate_next_run_time(config.schedule)
            
        except Exception as e:
            logger.error(f"Error running scheduled report {config.name}: {e}")
    
    def _calculate_next_run_time(self, schedule: str) -> datetime:
        """Calculate next run time from cron schedule"""
        # Simplified implementation - would use proper cron parser
        if schedule == 'daily':
            return datetime.utcnow() + timedelta(days=1)
        elif schedule == 'weekly':
            return datetime.utcnow() + timedelta(weeks=1)
        elif schedule == 'monthly':
            return datetime.utcnow() + timedelta(days=30)
        else:
            return datetime.utcnow() + timedelta(hours=1)
    
    async def get_dashboard_data(self, portfolio_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current dashboard data"""
        try:
            dashboard_data = {
                'metrics': {},
                'alerts': [],
                'charts': {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Filter metrics for specific portfolio if requested
            for metric_id, metric in self.dashboard_metrics.items():
                dashboard_data['metrics'][metric_id] = {
                    'name': metric.name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'status': metric.status,
                    'change_1h': metric.change_1h,
                    'change_24h': metric.change_24h,
                    'threshold': metric.threshold,
                    'timestamp': metric.timestamp.isoformat() if metric.timestamp else None
                }
            
            # Get active alerts
            if self.breach_detector:
                alerts = await self.breach_detector.get_active_alerts(portfolio_id)
                dashboard_data['alerts'] = alerts[:10]  # Limit to 10 most recent
            
            # Add chart data placeholders
            dashboard_data['charts'] = {
                'var_history': await self._get_var_history_chart_data(portfolio_id),
                'exposure_breakdown': await self._get_exposure_breakdown_chart_data(portfolio_id),
                'limit_utilization': await self._get_limit_utilization_chart_data(portfolio_id)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}
    
    async def _get_var_history_chart_data(self, portfolio_id: Optional[str]) -> Dict[str, Any]:
        """Get VaR history chart data"""
        # Mock chart data
        return {
            'labels': ['9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00'],
            'datasets': [
                {
                    'label': 'VaR 95%',
                    'data': [2500, 2600, 2450, 2700, 2550, 2800, 2650]
                },
                {
                    'label': 'VaR 99%',
                    'data': [3200, 3300, 3100, 3400, 3250, 3500, 3350]
                }
            ]
        }
    
    async def _get_exposure_breakdown_chart_data(self, portfolio_id: Optional[str]) -> Dict[str, Any]:
        """Get exposure breakdown chart data"""
        # Mock pie chart data
        return {
            'labels': ['Technology', 'Healthcare', 'Financial', 'Energy', 'Other'],
            'data': [45, 25, 15, 10, 5]
        }
    
    async def _get_limit_utilization_chart_data(self, portfolio_id: Optional[str]) -> Dict[str, Any]:
        """Get limit utilization chart data"""
        # Mock bar chart data
        return {
            'labels': ['VaR Limit', 'Concentration', 'Leverage', 'Position Size'],
            'data': [75, 60, 45, 30],
            'thresholds': [80, 80, 80, 80]
        }
    
    async def schedule_report(self, config: ReportConfig) -> str:
        """Schedule a recurring report"""
        try:
            config_id = f"report_{int(datetime.utcnow().timestamp())}"
            config.created_at = datetime.utcnow()
            config.next_run = self._calculate_next_run_time(config.schedule)
            
            self.report_configs[config_id] = config
            
            logger.info(f"Scheduled report: {config.name} ({config_id})")
            return config_id
            
        except Exception as e:
            logger.error(f"Error scheduling report: {e}")
            raise
    
    async def get_report_status(self) -> Dict[str, Any]:
        """Get reporting system status"""
        return {
            'dashboard_active': self._dashboard_task is not None and not self._dashboard_task.done(),
            'scheduler_active': self.scheduler_active,
            'scheduled_reports': len(self.report_configs),
            'cached_reports': len(self.report_cache),
            'dashboard_metrics': len(self.dashboard_metrics),
            'dashboard_update_interval': self.dashboard_update_interval,
            'cache_ttl_minutes': self.cache_ttl_minutes
        }

# Global instance
risk_reporter = RiskReporter()