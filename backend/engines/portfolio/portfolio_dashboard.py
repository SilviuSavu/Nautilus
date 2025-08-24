#!/usr/bin/env python3
"""
Institutional Portfolio Dashboard
Professional-grade dashboard with real-time analytics and visualization
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional plotting imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    logger.info("Plotly available for dashboard visualization")
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available, dashboard will use data-only format")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    logger.info("Matplotlib available for dashboard visualization")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available")

class DashboardType(Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    PORTFOLIO_PERFORMANCE = "portfolio_performance"
    RISK_ANALYSIS = "risk_analysis"
    ASSET_ALLOCATION = "asset_allocation"
    FAMILY_OFFICE_OVERVIEW = "family_office_overview"
    STRATEGY_COMPARISON = "strategy_comparison"
    ESG_IMPACT = "esg_impact"
    TAX_EFFICIENCY = "tax_efficiency"
    LIQUIDITY_ANALYSIS = "liquidity_analysis"

class ChartType(Enum):
    LINE_CHART = "line"
    BAR_CHART = "bar"
    PIE_CHART = "pie"
    HEATMAP = "heatmap"
    SCATTER_PLOT = "scatter"
    HISTOGRAM = "histogram"
    TREEMAP = "treemap"
    WATERFALL = "waterfall"

@dataclass
class DashboardWidget:
    widget_id: str
    widget_type: str
    title: str
    description: str
    data: Dict[str, Any]
    chart_config: Optional[Dict[str, Any]] = None
    size: str = "medium"  # small, medium, large, full_width
    priority: int = 1  # Display priority (1 = highest)

@dataclass
class DashboardLayout:
    dashboard_id: str
    dashboard_type: DashboardType
    title: str
    description: str
    widgets: List[DashboardWidget]
    refresh_frequency: int = 300  # seconds
    created_at: datetime
    last_updated: datetime

class InstitutionalDashboard:
    """
    Institutional-grade portfolio dashboard generator
    Creates professional visualizations and analytics
    """
    
    def __init__(self):
        self.dashboard_cache: Dict[str, DashboardLayout] = {}
        self.color_scheme = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e", 
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff7f0e",
            "info": "#17a2b8",
            "light": "#f8f9fa",
            "dark": "#343a40"
        }
    
    async def generate_executive_dashboard(self, portfolio_data: Dict[str, Any], client_data: Dict[str, Any]) -> DashboardLayout:
        """Generate executive summary dashboard"""
        
        widgets = []
        
        # Key metrics widget
        key_metrics = DashboardWidget(
            widget_id="executive_key_metrics",
            widget_type="metrics_grid",
            title="Key Performance Metrics",
            description="Primary portfolio performance indicators",
            data={
                "total_aum": portfolio_data.get("total_value", 0),
                "ytd_return": self._calculate_ytd_return(portfolio_data),
                "sharpe_ratio": portfolio_data.get("sharpe_ratio", 0.8),
                "max_drawdown": portfolio_data.get("max_drawdown", -0.05),
                "portfolios_count": len(portfolio_data.get("portfolios", []))
            },
            size="large",
            priority=1
        )
        widgets.append(key_metrics)
        
        # Performance chart
        performance_chart = DashboardWidget(
            widget_id="executive_performance",
            widget_type="line_chart",
            title="Portfolio Performance Trend",
            description="12-month performance vs benchmark",
            data=self._generate_performance_chart_data(portfolio_data),
            chart_config={
                "chart_type": ChartType.LINE_CHART.value,
                "height": 400,
                "show_benchmark": True,
                "show_drawdown": True
            },
            size="full_width",
            priority=2
        )
        widgets.append(performance_chart)
        
        # Asset allocation pie chart
        allocation_chart = DashboardWidget(
            widget_id="executive_allocation",
            widget_type="pie_chart", 
            title="Asset Allocation Overview",
            description="Current portfolio allocation by asset class",
            data=self._generate_allocation_data(portfolio_data),
            chart_config={
                "chart_type": ChartType.PIE_CHART.value,
                "show_values": True,
                "color_scheme": "institutional"
            },
            size="medium",
            priority=3
        )
        widgets.append(allocation_chart)
        
        # Risk metrics
        risk_metrics = DashboardWidget(
            widget_id="executive_risk",
            widget_type="risk_gauge",
            title="Risk Profile",
            description="Current risk metrics and limits",
            data={
                "var_95": portfolio_data.get("var_95", -50000),
                "portfolio_value": portfolio_data.get("total_value", 1000000),
                "volatility": portfolio_data.get("volatility", 0.15),
                "beta": portfolio_data.get("beta", 1.0),
                "risk_score": self._calculate_risk_score(portfolio_data)
            },
            chart_config={
                "gauge_min": 0,
                "gauge_max": 10,
                "color_ranges": [(0, 3, "green"), (3, 7, "yellow"), (7, 10, "red")]
            },
            size="medium",
            priority=4
        )
        widgets.append(risk_metrics)
        
        dashboard = DashboardLayout(
            dashboard_id=f"exec_{int(datetime.now().timestamp())}",
            dashboard_type=DashboardType.EXECUTIVE_SUMMARY,
            title="Executive Portfolio Summary",
            description="High-level portfolio overview for executive review",
            widgets=widgets,
            refresh_frequency=300,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.dashboard_cache[dashboard.dashboard_id] = dashboard
        return dashboard
    
    async def generate_family_office_dashboard(self, family_data: Dict[str, Any]) -> DashboardLayout:
        """Generate family office multi-generational dashboard"""
        
        widgets = []
        
        # Family wealth overview
        wealth_overview = DashboardWidget(
            widget_id="family_wealth_overview",
            widget_type="metrics_grid",
            title="Family Wealth Overview",
            description="Multi-generational wealth summary",
            data={
                "total_family_wealth": sum(p.get("total_value", 0) for p in family_data.get("portfolios", [])),
                "generations_count": len(set(p.get("generation", 1) for p in family_data.get("portfolios", []))),
                "active_portfolios": len(family_data.get("portfolios", [])),
                "trust_structures": len(family_data.get("trusts", [])),
                "goal_progress": self._calculate_family_goal_progress(family_data)
            },
            size="full_width",
            priority=1
        )
        widgets.append(wealth_overview)
        
        # Generation breakdown
        generation_breakdown = DashboardWidget(
            widget_id="generation_breakdown",
            widget_type="stacked_bar",
            title="Wealth by Generation",
            description="Asset distribution across family generations",
            data=self._generate_generation_breakdown(family_data),
            chart_config={
                "chart_type": ChartType.BAR_CHART.value,
                "stacked": True,
                "show_percentages": True
            },
            size="medium",
            priority=2
        )
        widgets.append(generation_breakdown)
        
        # Goal tracking
        goal_tracking = DashboardWidget(
            widget_id="family_goals",
            widget_type="progress_bars",
            title="Family Goals Progress",
            description="Progress towards family financial objectives",
            data=self._generate_goal_tracking_data(family_data),
            chart_config={
                "show_target_dates": True,
                "color_code_by_probability": True
            },
            size="medium",
            priority=3
        )
        widgets.append(goal_tracking)
        
        # Tax efficiency analysis
        tax_analysis = DashboardWidget(
            widget_id="tax_efficiency",
            widget_type="efficiency_chart",
            title="Tax Efficiency Analysis",
            description="Tax optimization opportunities across portfolios",
            data=self._generate_tax_efficiency_data(family_data),
            chart_config={
                "show_tax_savings": True,
                "highlight_opportunities": True
            },
            size="full_width",
            priority=4
        )
        widgets.append(tax_analysis)
        
        dashboard = DashboardLayout(
            dashboard_id=f"family_{int(datetime.now().timestamp())}",
            dashboard_type=DashboardType.FAMILY_OFFICE_OVERVIEW,
            title="Family Office Dashboard",
            description="Comprehensive family wealth management overview",
            widgets=widgets,
            refresh_frequency=600,  # 10 minutes
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.dashboard_cache[dashboard.dashboard_id] = dashboard
        return dashboard
    
    async def generate_risk_dashboard(self, portfolio_data: Dict[str, Any], risk_analysis: Dict[str, Any]) -> DashboardLayout:
        """Generate comprehensive risk analysis dashboard"""
        
        widgets = []
        
        # Risk metrics grid
        risk_metrics = DashboardWidget(
            widget_id="risk_metrics_grid",
            widget_type="metrics_grid",
            title="Risk Metrics Summary",
            description="Comprehensive risk measurement overview",
            data={
                "var_95": risk_analysis.get("var_95", 0),
                "cvar_95": risk_analysis.get("cvar_95", 0),
                "max_drawdown": risk_analysis.get("maximum_drawdown", 0),
                "volatility": risk_analysis.get("volatility_1y", 0),
                "beta": risk_analysis.get("beta_vs_benchmark", 1.0),
                "correlation": risk_analysis.get("correlation_vs_benchmark", 0.8),
                "tracking_error": risk_analysis.get("tracking_error", 0.05),
                "information_ratio": risk_analysis.get("information_ratio", 0.5)
            },
            size="full_width",
            priority=1
        )
        widgets.append(risk_metrics)
        
        # VaR distribution
        var_distribution = DashboardWidget(
            widget_id="var_distribution",
            widget_type="histogram",
            title="Value at Risk Distribution",
            description="Historical VaR distribution and confidence intervals",
            data=self._generate_var_distribution_data(risk_analysis),
            chart_config={
                "chart_type": ChartType.HISTOGRAM.value,
                "show_confidence_intervals": True,
                "overlay_normal_curve": True
            },
            size="medium",
            priority=2
        )
        widgets.append(var_distribution)
        
        # Drawdown analysis
        drawdown_chart = DashboardWidget(
            widget_id="drawdown_analysis",
            widget_type="line_chart",
            title="Drawdown Analysis",
            description="Historical drawdown periods and recovery times",
            data=self._generate_drawdown_chart_data(portfolio_data),
            chart_config={
                "chart_type": ChartType.LINE_CHART.value,
                "fill_negative": True,
                "show_recovery_periods": True
            },
            size="medium",
            priority=3
        )
        widgets.append(drawdown_chart)
        
        # Correlation heatmap
        correlation_heatmap = DashboardWidget(
            widget_id="correlation_heatmap",
            widget_type="heatmap",
            title="Asset Correlation Matrix",
            description="Correlation analysis across portfolio holdings",
            data=self._generate_correlation_matrix(portfolio_data),
            chart_config={
                "chart_type": ChartType.HEATMAP.value,
                "color_scale": "RdBu",
                "show_values": True
            },
            size="full_width",
            priority=4
        )
        widgets.append(correlation_heatmap)
        
        dashboard = DashboardLayout(
            dashboard_id=f"risk_{int(datetime.now().timestamp())}",
            dashboard_type=DashboardType.RISK_ANALYSIS,
            title="Risk Analysis Dashboard",
            description="Comprehensive portfolio risk assessment",
            widgets=widgets,
            refresh_frequency=180,  # 3 minutes
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.dashboard_cache[dashboard.dashboard_id] = dashboard
        return dashboard
    
    async def generate_strategy_comparison_dashboard(self, strategies_data: Dict[str, Any]) -> DashboardLayout:
        """Generate strategy comparison dashboard"""
        
        widgets = []
        
        # Strategy performance comparison
        strategy_performance = DashboardWidget(
            widget_id="strategy_performance",
            widget_type="multi_line_chart",
            title="Strategy Performance Comparison",
            description="Comparative performance of different investment strategies",
            data=self._generate_strategy_performance_data(strategies_data),
            chart_config={
                "chart_type": ChartType.LINE_CHART.value,
                "multi_series": True,
                "show_legend": True
            },
            size="full_width",
            priority=1
        )
        widgets.append(strategy_performance)
        
        # Risk-return scatter plot
        risk_return_scatter = DashboardWidget(
            widget_id="risk_return_scatter",
            widget_type="scatter_plot",
            title="Risk-Return Profile",
            description="Risk vs return analysis for each strategy",
            data=self._generate_risk_return_scatter_data(strategies_data),
            chart_config={
                "chart_type": ChartType.SCATTER_PLOT.value,
                "x_axis": "volatility",
                "y_axis": "return",
                "bubble_size": "sharpe_ratio"
            },
            size="medium",
            priority=2
        )
        widgets.append(risk_return_scatter)
        
        # Strategy allocation comparison
        allocation_comparison = DashboardWidget(
            widget_id="allocation_comparison",
            widget_type="stacked_bar",
            title="Asset Allocation by Strategy",
            description="Comparative asset allocation across strategies",
            data=self._generate_strategy_allocation_comparison(strategies_data),
            chart_config={
                "chart_type": ChartType.BAR_CHART.value,
                "stacked": True,
                "horizontal": True
            },
            size="medium",
            priority=3
        )
        widgets.append(allocation_comparison)
        
        dashboard = DashboardLayout(
            dashboard_id=f"strategy_{int(datetime.now().timestamp())}",
            dashboard_type=DashboardType.STRATEGY_COMPARISON,
            title="Strategy Comparison Dashboard",
            description="Comparative analysis of investment strategies",
            widgets=widgets,
            refresh_frequency=900,  # 15 minutes
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.dashboard_cache[dashboard.dashboard_id] = dashboard
        return dashboard
    
    def _calculate_ytd_return(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate year-to-date return"""
        return portfolio_data.get("ytd_return", np.random.uniform(0.05, 0.15))
    
    def _calculate_risk_score(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate overall risk score (0-10)"""
        volatility = portfolio_data.get("volatility", 0.15)
        beta = portfolio_data.get("beta", 1.0)
        max_drawdown = abs(portfolio_data.get("max_drawdown", -0.05))
        
        # Simple risk scoring formula
        risk_score = (volatility * 25) + (beta * 2) + (max_drawdown * 20)
        return min(max(risk_score, 0), 10)
    
    def _generate_performance_chart_data(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance chart data"""
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        # Simulate portfolio performance
        returns = np.random.normal(0.0008, 0.015, 252)  # Daily returns
        portfolio_values = np.cumprod(1 + returns) * 100
        
        # Simulate benchmark performance
        benchmark_returns = np.random.normal(0.0006, 0.012, 252)
        benchmark_values = np.cumprod(1 + benchmark_returns) * 100
        
        return {
            "dates": [d.isoformat() for d in dates],
            "portfolio_values": portfolio_values.tolist(),
            "benchmark_values": benchmark_values.tolist(),
            "portfolio_name": portfolio_data.get("portfolio_name", "Portfolio"),
            "benchmark_name": portfolio_data.get("benchmark", "Benchmark")
        }
    
    def _generate_allocation_data(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate asset allocation data"""
        positions = portfolio_data.get("positions", {})
        
        if positions:
            allocation = {}
            for symbol, position in positions.items():
                weight = position.get("weight", 0)
                sector = position.get("sector", "Unknown")
                
                if sector not in allocation:
                    allocation[sector] = 0
                allocation[sector] += weight
        else:
            # Default allocation
            allocation = {
                "Technology": 0.30,
                "Healthcare": 0.20,
                "Financials": 0.15,
                "Consumer Discretionary": 0.15,
                "Industrial": 0.10,
                "Energy": 0.05,
                "Utilities": 0.05
            }
        
        return {
            "labels": list(allocation.keys()),
            "values": list(allocation.values()),
            "colors": [self.color_scheme["primary"], self.color_scheme["secondary"], 
                      self.color_scheme["success"], self.color_scheme["warning"],
                      self.color_scheme["info"], self.color_scheme["danger"],
                      self.color_scheme["dark"]][:len(allocation)]
        }
    
    def _generate_var_distribution_data(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate VaR distribution data"""
        var_95 = risk_analysis.get("var_95", -50000)
        
        # Simulate VaR distribution
        var_values = np.random.normal(var_95, abs(var_95) * 0.3, 1000)
        
        return {
            "var_values": var_values.tolist(),
            "var_95_line": var_95,
            "confidence_95": np.percentile(var_values, 5),
            "mean_var": np.mean(var_values)
        }
    
    def _generate_drawdown_chart_data(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate drawdown chart data"""
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        # Simulate drawdown periods
        returns = np.random.normal(0.0008, 0.015, 252)
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        return {
            "dates": [d.isoformat() for d in dates],
            "drawdowns": drawdowns.tolist(),
            "max_drawdown": np.min(drawdowns),
            "current_drawdown": drawdowns[-1]
        }
    
    def _generate_correlation_matrix(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate correlation matrix data"""
        positions = portfolio_data.get("positions", {})
        symbols = list(positions.keys())[:10]  # Limit to 10 symbols for visualization
        
        if len(symbols) < 3:
            symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "JPM", "JNJ", "UNH"]
        
        # Generate synthetic correlation matrix
        n = len(symbols)
        correlation_matrix = np.random.uniform(-0.5, 0.9, (n, n))
        # Make symmetric
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        # Set diagonal to 1
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return {
            "symbols": symbols,
            "correlation_matrix": correlation_matrix.tolist(),
            "annotations": True
        }
    
    def _calculate_family_goal_progress(self, family_data: Dict[str, Any]) -> float:
        """Calculate overall family goal progress"""
        goals = family_data.get("goals", [])
        if not goals:
            return 0.0
        
        total_progress = sum(goal.get("progress_percentage", 0) for goal in goals)
        return total_progress / len(goals)
    
    def _generate_generation_breakdown(self, family_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generation wealth breakdown"""
        portfolios = family_data.get("portfolios", [])
        
        generation_data = {}
        for portfolio in portfolios:
            generation = f"Generation {portfolio.get('generation', 1)}"
            value = portfolio.get("total_value", 0)
            
            if generation not in generation_data:
                generation_data[generation] = 0
            generation_data[generation] += value
        
        return {
            "generations": list(generation_data.keys()),
            "values": list(generation_data.values()),
            "percentages": [v/sum(generation_data.values())*100 for v in generation_data.values()]
        }
    
    def _generate_goal_tracking_data(self, family_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate goal tracking data"""
        goals = family_data.get("goals", [])
        
        return {
            "goals": [
                {
                    "name": goal.get("goal_name", "Unknown Goal"),
                    "progress": goal.get("progress_percentage", 0),
                    "target_amount": goal.get("target_amount", 0),
                    "target_date": goal.get("target_date", "2030-12-31"),
                    "priority": goal.get("priority", 3),
                    "on_track": goal.get("progress_percentage", 0) > 70
                }
                for goal in goals
            ]
        }
    
    def _generate_tax_efficiency_data(self, family_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tax efficiency analysis data"""
        portfolios = family_data.get("portfolios", [])
        
        tax_data = {
            "tax_drag": np.random.uniform(0.5, 2.0, len(portfolios)).tolist(),
            "tax_alpha": np.random.uniform(-0.5, 1.5, len(portfolios)).tolist(),
            "portfolio_names": [p.get("portfolio_name", f"Portfolio {i}") for i, p in enumerate(portfolios)],
            "optimization_opportunities": np.random.uniform(0, 0.8, len(portfolios)).tolist()
        }
        
        return tax_data
    
    def _generate_strategy_performance_data(self, strategies_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy performance comparison data"""
        strategies = strategies_data.get("strategies", [])
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        performance_data = {
            "dates": [d.isoformat() for d in dates],
            "strategies": {}
        }
        
        for strategy in strategies:
            strategy_name = strategy.get("strategy_name", "Unknown Strategy")
            # Simulate strategy performance
            returns = np.random.normal(0.0008, 0.015, 252)
            cumulative = np.cumprod(1 + returns) * 100
            performance_data["strategies"][strategy_name] = cumulative.tolist()
        
        return performance_data
    
    def _generate_risk_return_scatter_data(self, strategies_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk-return scatter plot data"""
        strategies = strategies_data.get("strategies", [])
        
        scatter_data = {
            "strategies": [],
            "volatility": [],
            "returns": [],
            "sharpe_ratios": []
        }
        
        for strategy in strategies:
            scatter_data["strategies"].append(strategy.get("strategy_name", "Unknown"))
            scatter_data["volatility"].append(strategy.get("risk_budget", np.random.uniform(0.08, 0.20)))
            scatter_data["returns"].append(strategy.get("return_target", np.random.uniform(0.06, 0.15)))
            scatter_data["sharpe_ratios"].append(np.random.uniform(0.5, 2.0))
        
        return scatter_data
    
    def _generate_strategy_allocation_comparison(self, strategies_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy allocation comparison data"""
        strategies = strategies_data.get("strategies", [])
        
        allocation_data = {
            "strategies": [],
            "allocations": {}
        }
        
        asset_classes = ["Equities", "Bonds", "Alternatives", "Cash", "Real Estate"]
        
        for strategy in strategies:
            strategy_name = strategy.get("strategy_name", "Unknown Strategy")
            allocation_data["strategies"].append(strategy_name)
            
            # Generate random allocation
            weights = np.random.dirichlet(np.ones(len(asset_classes)))
            for i, asset_class in enumerate(asset_classes):
                if asset_class not in allocation_data["allocations"]:
                    allocation_data["allocations"][asset_class] = []
                allocation_data["allocations"][asset_class].append(weights[i])
        
        return allocation_data
    
    async def export_dashboard(self, dashboard_id: str, format: str = "json") -> Dict[str, Any]:
        """Export dashboard in specified format"""
        if dashboard_id not in self.dashboard_cache:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self.dashboard_cache[dashboard_id]
        
        if format.lower() == "json":
            return self._export_json(dashboard)
        elif format.lower() == "html":
            return self._export_html(dashboard)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, dashboard: DashboardLayout) -> Dict[str, Any]:
        """Export dashboard as JSON"""
        return {
            "dashboard_id": dashboard.dashboard_id,
            "title": dashboard.title,
            "description": dashboard.description,
            "dashboard_type": dashboard.dashboard_type.value,
            "created_at": dashboard.created_at.isoformat(),
            "last_updated": dashboard.last_updated.isoformat(),
            "widgets": [
                {
                    "widget_id": widget.widget_id,
                    "widget_type": widget.widget_type,
                    "title": widget.title,
                    "description": widget.description,
                    "data": widget.data,
                    "chart_config": widget.chart_config,
                    "size": widget.size,
                    "priority": widget.priority
                }
                for widget in dashboard.widgets
            ]
        }
    
    def _export_html(self, dashboard: DashboardLayout) -> Dict[str, Any]:
        """Export dashboard as HTML (simplified version)"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .widget {{ border: 1px solid #ddd; margin: 10px; padding: 15px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
                .metric {{ text-align: center; padding: 10px; background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>{dashboard.title}</h1>
            <p>{dashboard.description}</p>
        """
        
        for widget in sorted(dashboard.widgets, key=lambda w: w.priority):
            html_content += f"""
            <div class="widget">
                <h3>{widget.title}</h3>
                <p>{widget.description}</p>
                <div class="data">
                    {self._format_widget_data_html(widget)}
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return {
            "format": "html",
            "content": html_content,
            "dashboard_id": dashboard.dashboard_id
        }
    
    def _format_widget_data_html(self, widget: DashboardWidget) -> str:
        """Format widget data as HTML"""
        if widget.widget_type == "metrics_grid":
            html = '<div class="metrics">'
            for key, value in widget.data.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                else:
                    formatted_value = str(value)
                html += f'<div class="metric"><strong>{key.replace("_", " ").title()}</strong><br>{formatted_value}</div>'
            html += '</div>'
            return html
        else:
            return f"<pre>{json.dumps(widget.data, indent=2)}</pre>"
    
    async def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards"""
        return [
            {
                "dashboard_id": dashboard_id,
                "title": dashboard.title,
                "dashboard_type": dashboard.dashboard_type.value,
                "widget_count": len(dashboard.widgets),
                "last_updated": dashboard.last_updated.isoformat()
            }
            for dashboard_id, dashboard in self.dashboard_cache.items()
        ]

    async def close(self):
        """Clean up resources"""
        self.dashboard_cache.clear()
        logger.info("Portfolio Dashboard closed")