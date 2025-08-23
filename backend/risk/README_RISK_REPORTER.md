# Comprehensive Risk Reporter - Sprint 3

## Overview

The Comprehensive Risk Reporter is a advanced reporting system that provides multi-format risk reports, real-time dashboard data, and automated distribution capabilities for the Nautilus trading platform. It integrates seamlessly with existing risk management infrastructure including the limit engine, risk monitor, and analytics components.

## Key Features

### 🎯 **Multi-Format Reporting**
- **Executive Summaries**: High-level risk overview for management
- **Daily/Weekly/Monthly Reports**: Detailed risk analysis with trends
- **Regulatory Compliance**: Basel III and other regulatory frameworks
- **Stress Testing**: Scenario analysis and stress test results
- **VaR Backtesting**: Model validation and performance analysis
- **Breach Analysis**: Limit violation tracking and remediation
- **Correlation Analysis**: Portfolio correlation and diversification metrics
- **Concentration Analysis**: Position concentration and diversification

### 📊 **Dashboard Integration**
- **Real-time Metrics**: Live risk metrics for dashboard consumption
- **Interactive Widgets**: Configurable dashboard components
- **Chart Data**: Risk trends, exposure breakdowns, and utilization charts
- **Alert Integration**: Active risk alerts and breach notifications
- **Multi-Portfolio Views**: Aggregate and individual portfolio dashboards

### ⏰ **Automated Scheduling**
- **Flexible Scheduling**: Daily, weekly, monthly, or custom intervals
- **Automated Distribution**: Email and file-based report delivery
- **Template Management**: Customizable report templates
- **Retention Policies**: Configurable report archiving and cleanup

### 📈 **Export Capabilities**
- **Multiple Formats**: JSON, PDF, CSV, Excel, HTML
- **Template System**: Jinja2-based report templating
- **Visualization Ready**: Chart data for external visualization tools
- **API Integration**: RESTful API for external system integration

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Risk Reporter Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   Report        │    │   Dashboard     │    │   Export     │ │
│  │   Generation    │    │   Service       │    │   Service    │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                      │      │
│           └───────────────────────┼──────────────────────┘      │
│                                   │                             │
│  ┌─────────────────────────────────┼─────────────────────────┐   │
│  │              Core Engine        │                         │   │
│  │  ┌─────────────────┐    ┌──────▼──────┐    ┌──────────┐  │   │
│  │  │   Scheduler     │    │   Cache     │    │Template  │  │   │
│  │  │   Service       │    │   Manager   │    │ Manager  │  │   │
│  │  └─────────────────┘    └─────────────┘    └──────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                   │                             │
│  ┌─────────────────────────────────┼─────────────────────────┐   │
│  │         Integration Layer       │                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│   │
│  │  │Limit Engine │  │Risk Monitor │  │  Risk Analytics     ││   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘│   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Installation & Setup

### 1. Dependencies

The risk reporter requires the following dependencies:

```python
# Core dependencies
asyncio
logging
datetime
decimal
typing
dataclasses
enum
uuid

# Data processing
pandas
numpy
matplotlib
seaborn

# Templating and formatting
jinja2
base64
json
csv
io

# Integration
asyncpg  # Database
redis    # Caching and WebSocket
```

### 2. Initialization

```python
from risk.risk_reporter import ComprehensiveRiskReporter, init_risk_reporter

# Initialize the risk reporter
risk_reporter = await init_risk_reporter()

# Start background services
await risk_reporter.start_services()
```

### 3. Configuration

The risk reporter automatically integrates with existing components:
- **Limit Engine**: For limit utilization and breach data
- **Risk Monitor**: For real-time risk metrics and alerts
- **Risk Analytics**: For VaR calculations and exposure analysis
- **Database**: For persistent storage and historical data
- **Redis**: For real-time updates and caching

## Usage Examples

### Basic Report Generation

```python
from risk.risk_reporter import ReportType, ReportFormat

# Generate executive summary
report_data, metadata = await risk_reporter.generate_report(
    report_type=ReportType.EXECUTIVE_SUMMARY,
    portfolio_ids=["PORTFOLIO_001", "PORTFOLIO_002"],
    format=ReportFormat.JSON,
    parameters={"period": "Daily"}
)

print(f"Report generated in {metadata.generation_time_ms}ms")
print(f"Total VaR 95%: ${report_data['summary']['total_var_95']:,.2f}")
```

### Dashboard Data

```python
# Get real-time dashboard data
dashboard_data = await risk_reporter.get_dashboard_data(
    portfolio_ids=["PORTFOLIO_001", "PORTFOLIO_002"]
)

# Access summary metrics
total_value = dashboard_data['summary_metrics']['total_value']
active_alerts = dashboard_data['summary_metrics']['active_alerts']
risk_level = dashboard_data['summary_metrics']['risk_level']

# Access chart data
var_history = dashboard_data['charts']['var_history']
exposure_breakdown = dashboard_data['charts']['exposure_breakdown']
```

### Scheduled Reports

```python
from risk.risk_reporter import ReportConfig, ReportFrequency, ReportPriority

# Create scheduled report configuration
config = ReportConfig(
    report_id="daily_risk_executive",
    report_type=ReportType.DAILY_RISK,
    name="Daily Risk Report - Executive",
    description="Daily risk assessment for executive team",
    portfolio_ids=["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"],
    format=ReportFormat.PDF,
    frequency=ReportFrequency.DAILY,
    schedule_expression="0 8 * * *",  # Daily at 8 AM
    recipients=["cro@company.com", "risk-team@company.com"],
    priority=ReportPriority.HIGH,
    active=True
)

# Schedule the report
config_id = await risk_reporter.schedule_report(config)
print(f"Report scheduled with ID: {config_id}")
```

### Export Reports

```python
# Generate and export report in multiple formats
report_data, metadata = await risk_reporter.generate_report(
    report_type=ReportType.STRESS_TEST,
    portfolio_ids=["PORTFOLIO_001"],
    format=ReportFormat.JSON,
    parameters={"scenarios": ["MARKET_CRASH", "VOLATILITY_SPIKE"]}
)

# Export to Excel
excel_path = await risk_reporter.export_report(
    report_data, 
    ReportFormat.EXCEL, 
    "/tmp/stress_test_report.xlsx"
)

# Export to PDF
pdf_data = await risk_reporter.export_report(
    report_data, 
    ReportFormat.PDF
)  # Returns base64 encoded data
```

## Report Types

### 1. Executive Summary
High-level risk overview including:
- Portfolio values and risk metrics
- Top risks and recommendations
- Regulatory compliance status
- Key performance indicators

### 2. Daily Risk Report
Comprehensive daily analysis including:
- Portfolio risk metrics (VaR, exposure, concentration)
- Daily P&L and performance
- Risk alerts and limit status
- Market summary and context

### 3. Stress Test Report
Scenario analysis including:
- Multiple stress scenarios
- Portfolio impact assessment
- Recovery time estimates
- Risk mitigation recommendations

### 4. Regulatory Compliance Report
Compliance assessment including:
- Basel III metrics
- Capital adequacy ratios
- Leverage and liquidity ratios
- Violation analysis and remediation plans

### 5. VaR Backtest Report
Model validation including:
- VaR exception analysis
- Kupiec and Christoffersen tests
- Model adequacy assessment
- Recalibration recommendations

## Dashboard Widgets

The risk reporter provides several dashboard widget types:

### 1. Metric Widgets
- Total VaR across portfolios
- Active alert counts
- Limit utilization percentages
- Risk level indicators

### 2. Chart Widgets
- VaR history trends
- Exposure breakdowns
- Limit utilization bars
- Risk attribution charts

### 3. Table Widgets
- Portfolio summaries
- Top risk positions
- Limit breach details
- Alert lists

### 4. Gauge Widgets
- Risk level meters
- Limit utilization gauges
- Compliance scores
- Performance indicators

## API Integration

### REST Endpoints

The risk reporter integrates with existing API routes:

```python
# Generate report via API
POST /api/v1/risk/reports/generate
{
    "report_type": "executive_summary",
    "portfolio_ids": ["PORTFOLIO_001"],
    "format": "json",
    "parameters": {"period": "daily"}
}

# Get dashboard data
GET /api/v1/risk/dashboard/data?portfolio_ids=PORTFOLIO_001,PORTFOLIO_002

# Schedule report
POST /api/v1/risk/reports/schedule
{
    "report_type": "daily_risk",
    "name": "Daily Risk Report",
    "frequency": "daily",
    "portfolio_ids": ["PORTFOLIO_001"],
    "recipients": ["risk@company.com"]
}
```

### WebSocket Updates

Real-time dashboard updates are broadcast via WebSocket:

```javascript
// Subscribe to risk dashboard updates
ws.subscribe('risk_dashboard_updates', (data) => {
    updateDashboard(data.summary_metrics);
    updateCharts(data.charts);
    updateAlerts(data.alerts);
});
```

## Configuration Options

### Report Configuration

```python
# Comprehensive report configuration
config = ReportConfig(
    report_id="custom_report",
    report_type=ReportType.MONTHLY_RISK,
    name="Monthly Risk Assessment",
    description="Comprehensive monthly risk review",
    portfolio_ids=["PORT_001", "PORT_002"],
    format=ReportFormat.PDF,
    frequency=ReportFrequency.MONTHLY,
    schedule_expression="0 9 1 * *",  # First day of month at 9 AM
    recipients=["cro@company.com", "board@company.com"],
    parameters={
        "include_stress_tests": True,
        "include_var_backtest": True,
        "regulation": "Basel III"
    },
    template_id="executive_template",
    priority=ReportPriority.CRITICAL,
    retention_days=365,
    active=True
)
```

### Dashboard Configuration

```python
# Custom dashboard widget
widget = DashboardWidget(
    widget_id="portfolio_var_gauge",
    title="Portfolio VaR (95%)",
    widget_type="gauge",
    data_source="var_95_total",
    refresh_interval=30,
    parameters={"confidence_level": 0.95},
    visualization_config={
        "min": 0,
        "max": 100000,
        "thresholds": {"warning": 75000, "critical": 90000}
    },
    position={"x": 0, "y": 0, "width": 3, "height": 3},
    active=True
)
```

## Performance & Monitoring

### Caching Strategy

The risk reporter uses intelligent caching:
- **Report Cache**: 30-minute TTL for generated reports
- **Dashboard Cache**: 5-second refresh for real-time data
- **Template Cache**: Persistent template storage
- **Widget Cache**: Component-level caching

### Monitoring Metrics

```python
# Get reporter statistics
stats = risk_reporter.get_statistics()

print(f"Reports Generated: {stats['reports_generated']}")
print(f"Average Generation Time: {stats['avg_generation_time_ms']:.2f}ms")
print(f"Cache Hit Rate: {stats['cache_hits'] / stats['total_requests']:.2%}")
print(f"Error Rate: {stats['error_count'] / stats['reports_generated']:.2%}")
```

### Performance Optimization

- **Concurrent Report Generation**: Maximum 5 concurrent reports
- **Intelligent Caching**: Reduces redundant calculations
- **Lazy Loading**: Components loaded on demand
- **Background Processing**: Scheduled reports run asynchronously

## Error Handling & Logging

The risk reporter implements comprehensive error handling:

### Error Categories
1. **Generation Errors**: Report generation failures
2. **Integration Errors**: Component unavailability
3. **Export Errors**: Format conversion failures
4. **Scheduling Errors**: Background task failures

### Logging Levels
- **INFO**: Normal operations and statistics
- **WARNING**: Non-critical issues and fallbacks
- **ERROR**: Failed operations with recovery
- **CRITICAL**: System failures requiring intervention

## Integration with Existing Components

### Limit Engine Integration
```python
# Automatic integration with limit engine
if self.limit_engine:
    limit_summary = await self.limit_engine.get_portfolio_limit_summary(portfolio_id)
    utilization_data = limit_summary.get('limit_checks', [])
```

### Risk Monitor Integration
```python
# Real-time risk data from monitor
if self.risk_monitor:
    risk_data = await self.risk_monitor.get_real_time_risk(portfolio_id)
    alerts = await self.risk_monitor.check_risk_breaches(portfolio_id)
```

### Analytics Integration
```python
# Advanced analytics from risk analytics engine
if self.risk_analytics:
    var_result = await self.risk_analytics.calculate_var(portfolio_id)
    stress_result = await self.risk_analytics.run_stress_test(portfolio_id, scenario)
```

## Testing

Run the comprehensive test suite:

```bash
cd /backend/risk
python test_risk_reporter.py
```

The test suite covers:
- Basic report generation
- Multiple report formats
- Dashboard data generation
- Export functionality
- Scheduling system
- Integration testing
- Performance testing

## Future Enhancements

### Planned Features
1. **Advanced Analytics**: Machine learning-based risk forecasting
2. **Enhanced Visualizations**: Interactive charts and graphs
3. **Mobile Dashboard**: Mobile-optimized risk dashboard
4. **API Versioning**: Multiple API versions for backward compatibility
5. **Custom Metrics**: User-defined risk metrics and calculations

### Extension Points
The risk reporter is designed for extensibility:
- **Custom Report Types**: Add new report generators
- **Custom Widgets**: Create specialized dashboard components
- **Custom Exporters**: Support additional output formats
- **Custom Integrations**: Connect to external risk systems

## Support & Maintenance

### Troubleshooting

Common issues and solutions:

1. **Report Generation Slow**
   - Check database connection pool
   - Verify analytics component availability
   - Review cache configuration

2. **Dashboard Not Updating**
   - Verify WebSocket connection
   - Check Redis connectivity
   - Review background task status

3. **Export Failures**
   - Verify file permissions
   - Check available disk space
   - Review format-specific dependencies

### Maintenance Tasks

Regular maintenance includes:
- Cache cleanup and optimization
- Log rotation and archiving
- Performance metrics review
- Template and configuration updates

---

For technical support or feature requests, please contact the Risk Management team or file an issue in the project repository.