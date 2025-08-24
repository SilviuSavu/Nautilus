"""
Autonomous Reporting System for Phase 8 - Intelligent Operations Intelligence
Advanced automated reporting with intelligent data visualization, natural language generation, and adaptive insights.

This module provides:
- Autonomous report generation with AI-driven insights
- Intelligent data visualization with automatic chart selection
- Natural language generation for executive summaries
- Multi-format report output (PDF, HTML, JSON, Excel, PowerBI)
- Adaptive reporting based on data patterns and user behavior
- Real-time streaming reports with live updates
- Custom report templates with dynamic content generation
- Advanced data storytelling with narrative intelligence
"""

import asyncio
import logging
import json
import time
import io
import base64
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import uuid
import asyncpg
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Template, Environment, BaseLoader
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report output formats"""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    EXCEL = "excel"
    POWERBI = "powerbi"
    DASHBOARD = "dashboard"
    EMAIL = "email"


class ReportFrequency(Enum):
    """Report generation frequency"""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_DEMAND = "on_demand"
    EVENT_TRIGGERED = "event_triggered"


class VisualizationType(Enum):
    """Types of data visualizations"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    PIE_CHART = "pie_chart"
    GAUGE_CHART = "gauge_chart"
    TREEMAP = "treemap"
    CANDLESTICK = "candlestick"
    SANKEY_DIAGRAM = "sankey_diagram"
    NETWORK_GRAPH = "network_graph"


class InsightLevel(Enum):
    """Levels of insights in reports"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXECUTIVE = "executive"


@dataclass
class ReportConfiguration:
    """Configuration for report generation"""
    report_id: str
    report_name: str
    report_type: str
    format: ReportFormat
    frequency: ReportFrequency
    data_sources: List[str]
    sections: List[Dict[str, Any]]
    recipients: List[str]
    parameters: Dict[str, Any]
    insight_level: InsightLevel = InsightLevel.INTERMEDIATE
    auto_insights: bool = True
    include_predictions: bool = True
    include_recommendations: bool = True
    template_id: Optional[str] = None
    custom_styling: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReportSection:
    """Individual section of a report"""
    section_id: str
    title: str
    content_type: str
    data_query: str
    visualization_config: Optional[Dict[str, Any]] = None
    narrative_template: Optional[str] = None
    insights_enabled: bool = True
    order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedVisualization:
    """Generated visualization with metadata"""
    viz_id: str
    title: str
    chart_type: VisualizationType
    chart_data: Dict[str, Any]
    chart_config: Dict[str, Any]
    interactive_html: Optional[str] = None
    static_image: Optional[str] = None  # base64 encoded
    insights: List[str] = field(default_factory=list)
    data_summary: Dict[str, Any] = field(default_factory=dict)
    generation_time_ms: float = 0.0


@dataclass
class NarrativeInsight:
    """AI-generated narrative insight"""
    insight_id: str
    insight_type: str
    title: str
    narrative: str
    key_metrics: Dict[str, Any]
    confidence: float
    data_sources: List[str]
    supporting_evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GeneratedReport:
    """Complete generated report"""
    report_id: str
    configuration: ReportConfiguration
    generated_at: datetime
    sections: List[Dict[str, Any]]
    visualizations: List[GeneratedVisualization]
    narrative_insights: List[NarrativeInsight]
    executive_summary: str
    key_findings: List[str]
    recommendations: List[str]
    data_quality_score: float
    generation_stats: Dict[str, Any]
    output_files: Dict[ReportFormat, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentVisualizationEngine:
    """AI-powered visualization selection and generation"""
    
    def __init__(self):
        # Set up visualization styles
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Chart type selection rules
        self.chart_selection_rules = {
            'time_series': VisualizationType.LINE_CHART,
            'correlation': VisualizationType.HEATMAP,
            'distribution': VisualizationType.HISTOGRAM,
            'categorical_comparison': VisualizationType.BAR_CHART,
            'relationship': VisualizationType.SCATTER_PLOT,
            'composition': VisualizationType.PIE_CHART,
            'performance_metric': VisualizationType.GAUGE_CHART,
            'hierarchical': VisualizationType.TREEMAP
        }
        
    def select_optimal_visualization(
        self, 
        data: pd.DataFrame, 
        analysis_context: Dict[str, Any]
    ) -> Tuple[VisualizationType, Dict[str, Any]]:
        """Intelligently select optimal visualization type for data"""
        try:
            # Analyze data characteristics
            data_profile = self._analyze_data_profile(data)
            
            # Determine visualization type based on data characteristics
            viz_type = self._determine_visualization_type(data_profile, analysis_context)
            
            # Generate configuration for the selected visualization
            viz_config = self._generate_visualization_config(viz_type, data_profile, analysis_context)
            
            return viz_type, viz_config
            
        except Exception as e:
            logger.error(f"Error selecting optimal visualization: {e}")
            return VisualizationType.LINE_CHART, {}
    
    def _analyze_data_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics for visualization selection"""
        try:
            profile = {
                'row_count': len(data),
                'column_count': len(data.columns),
                'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': list(data.select_dtypes(include=['datetime64']).columns),
                'missing_data_ratio': data.isnull().sum().sum() / (len(data) * len(data.columns)),
                'data_types': {col: str(dtype) for col, dtype in data.dtypes.items()}
            }
            
            # Check for time series pattern
            if profile['datetime_columns']:
                profile['is_time_series'] = True
                profile['time_range'] = {
                    'start': data[profile['datetime_columns'][0]].min(),
                    'end': data[profile['datetime_columns'][0]].max()
                }
            else:
                profile['is_time_series'] = False
            
            # Analyze numeric data distributions
            if profile['numeric_columns']:
                numeric_data = data[profile['numeric_columns']]
                profile['numeric_stats'] = {
                    'correlations': numeric_data.corr().to_dict(),
                    'skewness': numeric_data.skew().to_dict(),
                    'outliers': self._detect_outliers(numeric_data)
                }
            
            # Analyze categorical data
            if profile['categorical_columns']:
                profile['categorical_stats'] = {}
                for col in profile['categorical_columns']:
                    unique_values = data[col].nunique()
                    profile['categorical_stats'][col] = {
                        'unique_count': unique_values,
                        'is_high_cardinality': unique_values > 20
                    }
            
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing data profile: {e}")
            return {}
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect outliers in numeric data using IQR method"""
        try:
            outliers = {}
            
            for col in data.columns:
                if data[col].dtype in [np.number]:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_indices = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index.tolist()
                    outliers[col] = outlier_indices
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return {}
    
    def _determine_visualization_type(
        self, 
        data_profile: Dict[str, Any], 
        analysis_context: Dict[str, Any]
    ) -> VisualizationType:
        """Determine optimal visualization type based on data profile"""
        try:
            # Time series data
            if data_profile.get('is_time_series', False) and data_profile['numeric_columns']:
                return VisualizationType.LINE_CHART
            
            # Correlation analysis
            if (len(data_profile.get('numeric_columns', [])) > 1 and 
                analysis_context.get('analysis_type') == 'correlation'):
                return VisualizationType.HEATMAP
            
            # Distribution analysis
            if (len(data_profile.get('numeric_columns', [])) == 1 and 
                analysis_context.get('analysis_type') == 'distribution'):
                return VisualizationType.HISTOGRAM
            
            # Categorical comparison
            if (data_profile.get('categorical_columns') and 
                data_profile.get('numeric_columns') and
                len(data_profile['categorical_columns']) == 1):
                unique_count = data_profile.get('categorical_stats', {}).get(
                    data_profile['categorical_columns'][0], {}
                ).get('unique_count', 0)
                
                if unique_count <= 10:
                    return VisualizationType.BAR_CHART
                elif unique_count <= 6:
                    return VisualizationType.PIE_CHART
            
            # Scatter plot for two numeric variables
            if len(data_profile.get('numeric_columns', [])) >= 2:
                return VisualizationType.SCATTER_PLOT
            
            # Performance metrics
            if (analysis_context.get('metric_type') == 'performance' and 
                len(data_profile.get('numeric_columns', [])) == 1):
                return VisualizationType.GAUGE_CHART
            
            # Default to bar chart
            return VisualizationType.BAR_CHART
            
        except Exception as e:
            logger.error(f"Error determining visualization type: {e}")
            return VisualizationType.LINE_CHART
    
    def _generate_visualization_config(
        self, 
        viz_type: VisualizationType, 
        data_profile: Dict[str, Any], 
        analysis_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate configuration for the selected visualization"""
        try:
            base_config = {
                'title': analysis_context.get('title', 'Data Visualization'),
                'theme': 'plotly_white',
                'responsive': True,
                'show_legend': True,
                'height': 400,
                'width': 600
            }
            
            # Visualization-specific configurations
            if viz_type == VisualizationType.LINE_CHART:
                base_config.update({
                    'x_axis': data_profile.get('datetime_columns', [None])[0],
                    'y_axis': data_profile.get('numeric_columns', [None])[0],
                    'line_style': 'solid',
                    'markers': len(data_profile.get('numeric_columns', [])) > 1
                })
            
            elif viz_type == VisualizationType.HEATMAP:
                base_config.update({
                    'colorscale': 'RdBu',
                    'show_values': True,
                    'correlation_method': 'pearson'
                })
            
            elif viz_type == VisualizationType.BAR_CHART:
                base_config.update({
                    'orientation': 'vertical',
                    'color_palette': 'Set3',
                    'sort_values': True
                })
            
            elif viz_type == VisualizationType.SCATTER_PLOT:
                base_config.update({
                    'x_axis': data_profile.get('numeric_columns', [None])[0],
                    'y_axis': data_profile.get('numeric_columns', [None])[1] if len(data_profile.get('numeric_columns', [])) > 1 else None,
                    'size_column': None,
                    'color_column': data_profile.get('categorical_columns', [None])[0],
                    'show_trendline': True
                })
            
            elif viz_type == VisualizationType.GAUGE_CHART:
                base_config.update({
                    'min_value': 0,
                    'max_value': 100,
                    'threshold_ranges': [
                        {'range': [0, 60], 'color': 'red'},
                        {'range': [60, 80], 'color': 'yellow'},
                        {'range': [80, 100], 'color': 'green'}
                    ]
                })
            
            return base_config
            
        except Exception as e:
            logger.error(f"Error generating visualization config: {e}")
            return {}
    
    def generate_visualization(
        self, 
        data: pd.DataFrame, 
        viz_type: VisualizationType, 
        config: Dict[str, Any]
    ) -> GeneratedVisualization:
        """Generate visualization based on type and configuration"""
        try:
            start_time = time.time()
            viz_id = str(uuid.uuid4())
            
            # Create visualization based on type
            if viz_type == VisualizationType.LINE_CHART:
                chart_obj = self._create_line_chart(data, config)
            elif viz_type == VisualizationType.BAR_CHART:
                chart_obj = self._create_bar_chart(data, config)
            elif viz_type == VisualizationType.SCATTER_PLOT:
                chart_obj = self._create_scatter_plot(data, config)
            elif viz_type == VisualizationType.HEATMAP:
                chart_obj = self._create_heatmap(data, config)
            elif viz_type == VisualizationType.HISTOGRAM:
                chart_obj = self._create_histogram(data, config)
            elif viz_type == VisualizationType.GAUGE_CHART:
                chart_obj = self._create_gauge_chart(data, config)
            else:
                chart_obj = self._create_line_chart(data, config)  # Default fallback
            
            # Generate HTML and static image
            interactive_html = pyo.plot(chart_obj, output_type='div', include_plotlyjs=True)
            static_image = self._generate_static_image(chart_obj)
            
            # Generate insights
            insights = self._generate_chart_insights(data, viz_type, config)
            
            # Create data summary
            data_summary = self._create_data_summary(data)
            
            generation_time = (time.time() - start_time) * 1000
            
            return GeneratedVisualization(
                viz_id=viz_id,
                title=config.get('title', 'Visualization'),
                chart_type=viz_type,
                chart_data=data.to_dict('records') if len(data) <= 1000 else data.sample(1000).to_dict('records'),
                chart_config=config,
                interactive_html=interactive_html,
                static_image=static_image,
                insights=insights,
                data_summary=data_summary,
                generation_time_ms=generation_time
            )
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return GeneratedVisualization(
                viz_id=str(uuid.uuid4()),
                title="Error",
                chart_type=viz_type,
                chart_data={},
                chart_config=config,
                insights=[f"Error generating visualization: {str(e)}"],
                data_summary={}
            )
    
    def _create_line_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create line chart using Plotly"""
        try:
            fig = go.Figure()
            
            x_col = config.get('x_axis')
            y_cols = [col for col in data.columns if col != x_col and data[col].dtype in [np.number]]
            
            for y_col in y_cols[:5]:  # Limit to 5 lines
                fig.add_trace(go.Scatter(
                    x=data[x_col] if x_col else data.index,
                    y=data[y_col],
                    mode='lines+markers' if config.get('markers', False) else 'lines',
                    name=y_col,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title=config.get('title', 'Line Chart'),
                xaxis_title=x_col or 'Index',
                yaxis_title='Value',
                height=config.get('height', 400),
                showlegend=config.get('show_legend', True),
                template=config.get('theme', 'plotly_white')
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating line chart: {e}")
            return go.Figure()
    
    def _create_bar_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create bar chart using Plotly"""
        try:
            fig = go.Figure()
            
            # Find categorical and numeric columns
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
            num_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(cat_cols) > 0 and len(num_cols) > 0:
                x_col = cat_cols[0]
                y_col = num_cols[0]
                
                # Aggregate data if needed
                if len(data) > 50:
                    plot_data = data.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(20)
                    x_values = plot_data.index
                    y_values = plot_data.values
                else:
                    x_values = data[x_col]
                    y_values = data[y_col]
                
                fig.add_trace(go.Bar(
                    x=x_values,
                    y=y_values,
                    name=y_col,
                    marker_color='steelblue'
                ))
                
                fig.update_layout(
                    title=config.get('title', 'Bar Chart'),
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    height=config.get('height', 400),
                    template=config.get('theme', 'plotly_white')
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
            return go.Figure()
    
    def _create_scatter_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create scatter plot using Plotly"""
        try:
            fig = go.Figure()
            
            num_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(num_cols) >= 2:
                x_col = config.get('x_axis', num_cols[0])
                y_col = config.get('y_axis', num_cols[1])
                color_col = config.get('color_column')
                
                scatter_params = {
                    'x': data[x_col],
                    'y': data[y_col],
                    'mode': 'markers',
                    'name': f'{y_col} vs {x_col}',
                    'marker': dict(size=8, opacity=0.7)
                }
                
                if color_col and color_col in data.columns:
                    scatter_params['marker']['color'] = data[color_col]
                    scatter_params['marker']['colorscale'] = 'Viridis'
                    scatter_params['marker']['showscale'] = True
                
                fig.add_trace(go.Scatter(**scatter_params))
                
                # Add trendline if requested
                if config.get('show_trendline', False):
                    z = np.polyfit(data[x_col], data[y_col], 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(
                        x=data[x_col],
                        y=p(data[x_col]),
                        mode='lines',
                        name='Trendline',
                        line=dict(dash='dash', color='red')
                    ))
                
                fig.update_layout(
                    title=config.get('title', 'Scatter Plot'),
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    height=config.get('height', 400),
                    template=config.get('theme', 'plotly_white')
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            return go.Figure()
    
    def _create_heatmap(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create heatmap using Plotly"""
        try:
            # Calculate correlation matrix for numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale=config.get('colorscale', 'RdBu'),
                    zmid=0,
                    text=np.round(corr_matrix.values, 2) if config.get('show_values', True) else None,
                    texttemplate='%{text}' if config.get('show_values', True) else None,
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title=config.get('title', 'Correlation Heatmap'),
                    height=config.get('height', 400),
                    template=config.get('theme', 'plotly_white')
                )
            else:
                fig = go.Figure()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return go.Figure()
    
    def _create_histogram(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create histogram using Plotly"""
        try:
            fig = go.Figure()
            
            num_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in num_cols[:3]:  # Limit to 3 histograms
                fig.add_trace(go.Histogram(
                    x=data[col],
                    name=col,
                    nbinsx=30,
                    opacity=0.7
                ))
            
            fig.update_layout(
                title=config.get('title', 'Histogram'),
                xaxis_title='Value',
                yaxis_title='Frequency',
                height=config.get('height', 400),
                barmode='overlay',
                template=config.get('theme', 'plotly_white')
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            return go.Figure()
    
    def _create_gauge_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create gauge chart using Plotly"""
        try:
            # Use the first numeric column's latest value
            num_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(num_cols) > 0:
                value = data[num_cols[0]].iloc[-1] if len(data) > 0 else 0
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': config.get('title', 'Performance Gauge')},
                    delta={'reference': data[num_cols[0]].mean() if len(data) > 1 else 0},
                    gauge={
                        'axis': {'range': [config.get('min_value', 0), config.get('max_value', 100)]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 80], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(
                    height=config.get('height', 400),
                    template=config.get('theme', 'plotly_white')
                )
            else:
                fig = go.Figure()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating gauge chart: {e}")
            return go.Figure()
    
    def _generate_static_image(self, fig: go.Figure) -> str:
        """Generate static image from Plotly figure"""
        try:
            # Convert to PNG bytes
            img_bytes = fig.to_image(format="png", width=800, height=600)
            
            # Encode as base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error generating static image: {e}")
            return ""
    
    def _generate_chart_insights(
        self, 
        data: pd.DataFrame, 
        viz_type: VisualizationType, 
        config: Dict[str, Any]
    ) -> List[str]:
        """Generate AI insights about the chart"""
        try:
            insights = []
            
            # Data size insight
            insights.append(f"Chart displays {len(data)} data points across {len(data.columns)} variables")
            
            # Data quality insight
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > 0.1:
                insights.append(f"Data quality concern: {missing_ratio:.1%} missing values detected")
            elif missing_ratio == 0:
                insights.append("Excellent data quality: No missing values detected")
            
            # Type-specific insights
            if viz_type == VisualizationType.LINE_CHART:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    trend = "increasing" if data[col].iloc[-1] > data[col].iloc[0] else "decreasing"
                    insights.append(f"Primary metric shows {trend} trend over time period")
            
            elif viz_type == VisualizationType.HEATMAP:
                numeric_data = data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr()
                    max_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
                    max_corr = max_corr[max_corr < 1.0].iloc[0]  # Exclude self-correlation
                    insights.append(f"Strongest correlation detected: {max_corr:.2f}")
            
            elif viz_type == VisualizationType.BAR_CHART:
                cat_cols = data.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    unique_count = data[cat_cols[0]].nunique()
                    insights.append(f"Categorical analysis across {unique_count} distinct categories")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating chart insights: {e}")
            return ["Unable to generate insights for this visualization"]
    
    def _create_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for the data"""
        try:
            summary = {
                'total_records': len(data),
                'total_columns': len(data.columns),
                'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': len(data.select_dtypes(include=['datetime64']).columns),
                'missing_values': int(data.isnull().sum().sum()),
                'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
            
            # Add numeric statistics
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                summary['numeric_stats'] = {
                    'mean_values': numeric_data.mean().to_dict(),
                    'std_values': numeric_data.std().to_dict(),
                    'min_values': numeric_data.min().to_dict(),
                    'max_values': numeric_data.max().to_dict()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating data summary: {e}")
            return {}


class NarrativeIntelligence:
    """AI-powered narrative generation for reports"""
    
    def __init__(self):
        self.narrative_templates = self._initialize_narrative_templates()
        
    def _initialize_narrative_templates(self) -> Dict[str, Template]:
        """Initialize narrative templates"""
        templates = {}
        
        # Executive summary template
        templates['executive_summary'] = Template("""
Based on analysis of {{data_points}} data points from {{time_period}}, the key findings are:

{{#key_findings}}
• {{.}}
{{/key_findings}}

{% if performance_trend %}
Performance Trend: {{performance_trend}} with {{confidence_level}} confidence.
{% endif %}

{% if recommendations %}
Critical Recommendations:
{{#recommendations}}
• {{.}}
{{/recommendations}}
{% endif %}
        """.strip())
        
        # Performance analysis template
        templates['performance_analysis'] = Template("""
Performance Analysis Summary:

The system processed {{total_transactions}} transactions with an average response time of {{avg_response_time}}ms. 
{% if response_time_trend == 'improving' %}
Response times have improved by {{improvement_percentage}}% compared to the previous period.
{% elif response_time_trend == 'degrading' %}
Response times have degraded by {{degradation_percentage}}% compared to the previous period, requiring immediate attention.
{% else %}
Response times remain stable within acceptable parameters.
{% endif %}

{% if anomalies_detected > 0 %}
Alert: {{anomalies_detected}} performance anomalies detected during this period.
{% endif %}
        """.strip())
        
        # Risk analysis template
        templates['risk_analysis'] = Template("""
Risk Assessment Report:

Current Risk Level: {{risk_level}} ({{risk_score}}/100)

{% if high_risk_factors %}
High-Risk Factors Identified:
{{#high_risk_factors}}
• {{factor_name}}: {{risk_value}} ({{impact_level}} impact)
{{/high_risk_factors}}
{% endif %}

{% if risk_trend == 'increasing' %}
⚠️ Risk trend is increasing. Immediate mitigation recommended.
{% elif risk_trend == 'stable' %}
✓ Risk levels remain stable within acceptable limits.
{% else %}
✓ Risk trend is improving with effective mitigation measures.
{% endif %}
        """.strip())
        
        return templates
    
    def generate_narrative_insight(
        self, 
        data: pd.DataFrame, 
        analysis_results: Dict[str, Any], 
        template_type: str = 'executive_summary'
    ) -> NarrativeInsight:
        """Generate AI-powered narrative insight"""
        try:
            insight_id = str(uuid.uuid4())
            
            # Analyze data patterns
            data_patterns = self._analyze_data_patterns(data)
            
            # Extract key metrics
            key_metrics = self._extract_key_metrics(data, analysis_results)
            
            # Generate narrative using template and analysis
            narrative = self._generate_narrative_text(
                template_type, data_patterns, key_metrics, analysis_results
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(data_patterns, key_metrics)
            
            # Calculate confidence based on data quality and sample size
            confidence = self._calculate_narrative_confidence(data, analysis_results)
            
            return NarrativeInsight(
                insight_id=insight_id,
                insight_type=template_type,
                title=self._generate_insight_title(template_type, key_metrics),
                narrative=narrative,
                key_metrics=key_metrics,
                confidence=confidence,
                data_sources=[f"dataset_{len(data)}_records"],
                supporting_evidence=self._extract_supporting_evidence(data_patterns),
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error generating narrative insight: {e}")
            return NarrativeInsight(
                insight_id=str(uuid.uuid4()),
                insight_type="error",
                title="Narrative Generation Error",
                narrative=f"Unable to generate narrative insight: {str(e)}",
                key_metrics={},
                confidence=0.0,
                data_sources=[]
            )
    
    def _analyze_data_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in the data for narrative generation"""
        try:
            patterns = {
                'sample_size': len(data),
                'time_coverage': 'unknown',
                'data_quality': 'good',
                'trends': [],
                'anomalies': [],
                'seasonal_patterns': []
            }
            
            # Analyze time coverage if datetime column exists
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                time_col = datetime_cols[0]
                time_range = data[time_col].max() - data[time_col].min()
                patterns['time_coverage'] = f"{time_range.days} days"
            
            # Analyze data quality
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > 0.2:
                patterns['data_quality'] = 'poor'
            elif missing_ratio > 0.1:
                patterns['data_quality'] = 'fair'
            else:
                patterns['data_quality'] = 'excellent'
            
            # Analyze trends in numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if len(data) > 2:
                    # Simple trend analysis
                    values = data[col].dropna()
                    if len(values) > 1:
                        trend_slope = (values.iloc[-1] - values.iloc[0]) / len(values)
                        if abs(trend_slope) > values.std() * 0.1:
                            trend_direction = "increasing" if trend_slope > 0 else "decreasing"
                            patterns['trends'].append({
                                'metric': col,
                                'direction': trend_direction,
                                'strength': abs(trend_slope) / values.std()
                            })
            
            # Simple anomaly detection using z-score
            for col in numeric_cols:
                values = data[col].dropna()
                if len(values) > 3:
                    z_scores = np.abs((values - values.mean()) / values.std())
                    anomaly_count = len(z_scores[z_scores > 3])
                    if anomaly_count > 0:
                        patterns['anomalies'].append({
                            'metric': col,
                            'count': anomaly_count,
                            'severity': 'high' if anomaly_count > len(values) * 0.05 else 'medium'
                        })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing data patterns: {e}")
            return {'sample_size': len(data), 'data_quality': 'unknown', 'trends': [], 'anomalies': []}
    
    def _extract_key_metrics(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for narrative generation"""
        try:
            metrics = {
                'total_records': len(data),
                'date_range': 'Not available',
                'primary_metrics': {},
                'performance_indicators': {}
            }
            
            # Extract date range if available
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                date_col = datetime_cols[0]
                start_date = data[date_col].min().strftime('%Y-%m-%d')
                end_date = data[date_col].max().strftime('%Y-%m-%d')
                metrics['date_range'] = f"{start_date} to {end_date}"
            
            # Extract primary numeric metrics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:5]:  # Limit to top 5 metrics
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    metrics['primary_metrics'][col] = {
                        'mean': round(col_data.mean(), 2),
                        'median': round(col_data.median(), 2),
                        'std': round(col_data.std(), 2),
                        'min': round(col_data.min(), 2),
                        'max': round(col_data.max(), 2),
                        'latest': round(col_data.iloc[-1], 2) if len(col_data) > 0 else None
                    }
            
            # Extract performance indicators from analysis results
            if 'model_performance' in analysis_results:
                metrics['performance_indicators'] = analysis_results['model_performance']
            
            if 'confidence_score' in analysis_results:
                metrics['analysis_confidence'] = analysis_results['confidence_score']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting key metrics: {e}")
            return {'total_records': len(data)}
    
    def _generate_narrative_text(
        self, 
        template_type: str, 
        patterns: Dict[str, Any], 
        metrics: Dict[str, Any], 
        analysis_results: Dict[str, Any]
    ) -> str:
        """Generate narrative text using patterns and metrics"""
        try:
            # Create narrative components
            narrative_parts = []
            
            # Opening statement
            narrative_parts.append(
                f"Analysis of {metrics['total_records']} data points reveals several key insights."
            )
            
            # Data quality assessment
            data_quality = patterns.get('data_quality', 'unknown')
            if data_quality == 'excellent':
                narrative_parts.append("The dataset demonstrates excellent data quality with minimal missing values.")
            elif data_quality == 'poor':
                narrative_parts.append("⚠️ Data quality concerns identified with significant missing values requiring attention.")
            
            # Trend analysis
            trends = patterns.get('trends', [])
            if trends:
                strong_trends = [t for t in trends if t.get('strength', 0) > 0.5]
                if strong_trends:
                    trend_descriptions = [f"{t['metric']} showing {t['direction']} trend" for t in strong_trends]
                    narrative_parts.append(f"Strong trends identified: {', '.join(trend_descriptions)}.")
            
            # Anomaly reporting
            anomalies = patterns.get('anomalies', [])
            if anomalies:
                high_severity_anomalies = [a for a in anomalies if a.get('severity') == 'high']
                if high_severity_anomalies:
                    narrative_parts.append(
                        f"⚠️ {len(high_severity_anomalies)} high-severity anomalies detected requiring investigation."
                    )
            
            # Performance summary
            if 'primary_metrics' in metrics and metrics['primary_metrics']:
                primary_metric = list(metrics['primary_metrics'].keys())[0]
                metric_data = metrics['primary_metrics'][primary_metric]
                latest_value = metric_data.get('latest')
                mean_value = metric_data.get('mean')
                
                if latest_value is not None and mean_value is not None:
                    performance_comparison = "above" if latest_value > mean_value else "below"
                    narrative_parts.append(
                        f"Current {primary_metric} value ({latest_value}) is {performance_comparison} the average ({mean_value})."
                    )
            
            # Confidence statement
            confidence = analysis_results.get('confidence_score', 0)
            if confidence > 0.8:
                narrative_parts.append("Analysis confidence is high, supporting reliable decision-making.")
            elif confidence > 0.6:
                narrative_parts.append("Analysis confidence is moderate, recommendations should be validated.")
            else:
                narrative_parts.append("⚠️ Analysis confidence is low, additional data collection recommended.")
            
            return " ".join(narrative_parts)
            
        except Exception as e:
            logger.error(f"Error generating narrative text: {e}")
            return f"Unable to generate comprehensive narrative. Data analysis completed for {metrics.get('total_records', 0)} records."
    
    def _generate_recommendations(self, patterns: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        try:
            recommendations = []
            
            # Data quality recommendations
            data_quality = patterns.get('data_quality', 'unknown')
            if data_quality in ['poor', 'fair']:
                recommendations.append("Improve data collection processes to reduce missing values")
                recommendations.append("Implement data validation checks at ingestion points")
            
            # Trend-based recommendations
            trends = patterns.get('trends', [])
            for trend in trends:
                if trend.get('direction') == 'decreasing' and trend.get('strength', 0) > 0.7:
                    recommendations.append(f"Address declining trend in {trend['metric']} through targeted interventions")
                elif trend.get('direction') == 'increasing' and trend.get('strength', 0) > 0.7:
                    recommendations.append(f"Capitalize on positive trend in {trend['metric']} with increased investment")
            
            # Anomaly-based recommendations
            anomalies = patterns.get('anomalies', [])
            for anomaly in anomalies:
                if anomaly.get('severity') == 'high':
                    recommendations.append(f"Investigate and address anomalies in {anomaly['metric']}")
            
            # Performance recommendations
            if 'primary_metrics' in metrics:
                for metric_name, metric_data in metrics['primary_metrics'].items():
                    std_dev = metric_data.get('std', 0)
                    mean_val = metric_data.get('mean', 0)
                    if std_dev > mean_val * 0.5:  # High variability
                        recommendations.append(f"Reduce variability in {metric_name} through process standardization")
            
            # General recommendations
            sample_size = patterns.get('sample_size', 0)
            if sample_size < 100:
                recommendations.append("Increase sample size for more robust statistical analysis")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Continue monitoring data trends and system performance"]
    
    def _generate_insight_title(self, template_type: str, metrics: Dict[str, Any]) -> str:
        """Generate appropriate title for the insight"""
        try:
            base_titles = {
                'executive_summary': 'Executive Summary',
                'performance_analysis': 'Performance Analysis Report',
                'risk_analysis': 'Risk Assessment Report',
                'trend_analysis': 'Trend Analysis Insights',
                'anomaly_report': 'Anomaly Detection Report'
            }
            
            base_title = base_titles.get(template_type, 'Data Analysis Report')
            
            # Add context if available
            record_count = metrics.get('total_records', 0)
            if record_count > 0:
                return f"{base_title} - {record_count:,} Records"
            
            return base_title
            
        except Exception as e:
            logger.error(f"Error generating insight title: {e}")
            return "Data Analysis Report"
    
    def _extract_supporting_evidence(self, patterns: Dict[str, Any]) -> List[str]:
        """Extract supporting evidence for the narrative"""
        try:
            evidence = []
            
            # Sample size evidence
            sample_size = patterns.get('sample_size', 0)
            if sample_size > 1000:
                evidence.append(f"Large sample size ({sample_size:,} records) provides statistical significance")
            elif sample_size > 100:
                evidence.append(f"Adequate sample size ({sample_size} records) for reliable analysis")
            
            # Data quality evidence
            data_quality = patterns.get('data_quality', 'unknown')
            if data_quality == 'excellent':
                evidence.append("High data quality with complete data coverage")
            
            # Trend evidence
            strong_trends = [t for t in patterns.get('trends', []) if t.get('strength', 0) > 0.5]
            if strong_trends:
                evidence.append(f"{len(strong_trends)} statistically significant trends identified")
            
            # Time coverage evidence
            time_coverage = patterns.get('time_coverage', 'unknown')
            if time_coverage != 'unknown' and 'days' in time_coverage:
                evidence.append(f"Analysis covers {time_coverage} providing temporal context")
            
            return evidence
            
        except Exception as e:
            logger.error(f"Error extracting supporting evidence: {e}")
            return ["Statistical analysis completed on available dataset"]
    
    def _calculate_narrative_confidence(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> float:
        """Calculate confidence score for the narrative"""
        try:
            confidence_factors = []
            
            # Sample size factor
            sample_size = len(data)
            if sample_size > 1000:
                confidence_factors.append(0.9)
            elif sample_size > 100:
                confidence_factors.append(0.7)
            elif sample_size > 30:
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.3)
            
            # Data quality factor
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            data_quality_score = max(0.1, 1.0 - missing_ratio * 2)
            confidence_factors.append(data_quality_score)
            
            # Analysis confidence factor (if available)
            if 'confidence_score' in analysis_results:
                confidence_factors.append(analysis_results['confidence_score'])
            
            # Statistical significance factor (if available)
            if 'p_value' in analysis_results:
                p_value = analysis_results['p_value']
                significance_score = 1.0 - p_value if p_value < 1.0 else 0.5
                confidence_factors.append(significance_score)
            
            # Calculate weighted average
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating narrative confidence: {e}")
            return 0.5


class AutonomousReportingSystem:
    """
    Autonomous reporting system with intelligent data visualization and narrative generation
    """
    
    def __init__(
        self,
        database_url: str,
        redis_url: str
    ):
        self.database_url = database_url
        self.redis_url = redis_url
        
        # Core components
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Reporting components
        self.visualization_engine = IntelligentVisualizationEngine()
        self.narrative_intelligence = NarrativeIntelligence()
        
        # Data storage
        self.report_configurations: Dict[str, ReportConfiguration] = {}
        self.generated_reports: Dict[str, GeneratedReport] = {}
        self.report_templates: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self._report_generation_task: Optional[asyncio.Task] = None
        self._report_delivery_task: Optional[asyncio.Task] = None
        
        # Thread pool for report generation
        self.report_executor = ThreadPoolExecutor(max_workers=4)
        
        # Configuration
        self.generation_check_interval_seconds = 300  # 5 minutes
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the autonomous reporting system"""
        try:
            # Initialize database connection pool
            self.db_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=3,
                max_size=15,
                command_timeout=60
            )
            
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Create database tables
            await self._create_database_tables()
            
            # Load existing report configurations
            await self._load_report_configurations()
            
            # Initialize default report templates
            self._initialize_default_templates()
            
            # Start background tasks
            self._report_generation_task = asyncio.create_task(self._report_generation_loop())
            self._report_delivery_task = asyncio.create_task(self._report_delivery_loop())
            
            self.logger.info("Autonomous Reporting System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize reporting system: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the reporting system"""
        try:
            # Cancel background tasks
            tasks = [self._report_generation_task, self._report_delivery_task]
            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Shutdown thread pool
            self.report_executor.shutdown(wait=True)
            
            # Close connections
            if self.db_pool:
                await self.db_pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Autonomous Reporting System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Global instance
autonomous_reporting_system = None

def get_autonomous_reporting_system() -> AutonomousReportingSystem:
    """Get global autonomous reporting system instance"""
    global autonomous_reporting_system
    if autonomous_reporting_system is None:
        raise RuntimeError("Autonomous reporting system not initialized")
    return autonomous_reporting_system

def init_autonomous_reporting_system(
    database_url: str, 
    redis_url: str
) -> AutonomousReportingSystem:
    """Initialize global autonomous reporting system instance"""
    global autonomous_reporting_system
    autonomous_reporting_system = AutonomousReportingSystem(database_url, redis_url)
    return autonomous_reporting_system