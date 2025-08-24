# API Reference

## Interactive Brokers (IBKR)
- `/api/v1/market-data/historical/bars` - Historical data from IBKR
- `/api/v1/ib/backfill` - Manual historical data backfill via IB Gateway
- `/api/v1/historical/backfill/status` - Backfill operation status
- `/api/v1/historical/backfill/stop` - Stop running backfill operations

## Alpha Vantage
- `/api/v1/alpha-vantage/health` - Integration health check
- `/api/v1/alpha-vantage/quote/{symbol}` - Real-time stock quotes
- `/api/v1/alpha-vantage/daily/{symbol}` - Daily historical data
- `/api/v1/alpha-vantage/intraday/{symbol}` - Intraday data (1min-60min)
- `/api/v1/alpha-vantage/search` - Symbol search by keywords
- `/api/v1/alpha-vantage/company/{symbol}` - Company fundamental data
- `/api/v1/alpha-vantage/earnings/{symbol}` - Quarterly/annual earnings
- `/api/v1/alpha-vantage/supported-functions` - List available functions

## FRED Economic Data
- `/api/v1/fred/health` - FRED API health check
- `/api/v1/fred/series` - List all 32+ available economic series
- `/api/v1/fred/series/{series_id}` - Get time series data for specific indicator
- `/api/v1/fred/series/{series_id}/latest` - Get latest value for economic series
- `/api/v1/fred/macro-factors` - Calculate institutional macro factors
- `/api/v1/fred/economic-calendar` - Economic release calendar
- `/api/v1/fred/cache/refresh` - Refresh economic data cache

## EDGAR SEC Filing Data
- `/api/v1/edgar/health` - EDGAR API health check
- `/api/v1/edgar/companies/search` - Search companies by name/ticker
- `/api/v1/edgar/companies/{cik}/facts` - Get company financial facts
- `/api/v1/edgar/companies/{cik}/filings` - Get recent company filings
- `/api/v1/edgar/ticker/{ticker}/resolve` - Resolve ticker to CIK and company name
- `/api/v1/edgar/ticker/{ticker}/facts` - Get financial facts by ticker
- `/api/v1/edgar/ticker/{ticker}/filings` - Get filings by ticker
- `/api/v1/edgar/filing-types` - List supported SEC form types
- `/api/v1/edgar/statistics` - EDGAR service statistics

## Data.gov Federal Datasets ⭐ **M4 MAX ACCELERATED** (Neural Engine + MessageBus)
**🏛️ 346,000+ U.S. Government datasets with M4 Max relevance scoring and Neural Engine analysis**
- `/api/v1/datagov/health` - Data.gov service health with M4 Max processing status
- `/api/v1/datagov/datasets/search` - M4 Max accelerated dataset search with Neural Engine relevance scoring
- `/api/v1/datagov/datasets/{id}` - Dataset details with M4 Max metadata processing
- `/api/v1/datagov/datasets/trading-relevant` - Neural Engine powered trading relevance scoring
- `/api/v1/datagov/categories` - 11 dataset categories with M4 Max classification
- `/api/v1/datagov/organizations` - Government agency listings with unified memory caching
- `/api/v1/datagov/datasets/category/{category}` - M4 Max optimized category filtering
- `/api/v1/datagov/datasets/load` - Hardware-accelerated dataset catalog loading

### MessageBus Data.gov Integration ⭐ **EVENT-DRIVEN ARCHITECTURE**
**🔄 Event-driven Data.gov access via Redis MessageBus**
- `/api/v1/datagov-mb/health` - MessageBus-enabled Data.gov health check
- `/api/v1/datagov-mb/datasets/search` - Event-driven dataset search
- `/api/v1/datagov-mb/datasets/{id}` - MessageBus dataset retrieval
- `/api/v1/datagov-mb/categories` - Categories via MessageBus
- `/api/v1/datagov-mb/status` - MessageBus service status and metrics

## DBnomics Economic Data ⭐ **EVENT-DRIVEN ARCHITECTURE** (MessageBus + REST)
**🏦 800M+ economic time series from 80+ official providers worldwide**
- `/api/v1/dbnomics/health` - DBnomics service health check with API availability
- `/api/v1/dbnomics/providers` - List of 80+ official data providers (IMF, OECD, ECB, etc.)
- `/api/v1/dbnomics/providers/{provider_code}/datasets` - Datasets for specific provider
- `/api/v1/dbnomics/series` - Search economic time series with filters
- `/api/v1/dbnomics/series/{provider}/{dataset}/{series}` - Get specific time series data
- `/api/v1/dbnomics/statistics` - Platform statistics and provider rankings
- `/api/v1/dbnomics/series/search` - Complex search via POST with dimensions

## NautilusTrader Engine Management (M4 Max Optimized) 🏆 **PRODUCTION READY**
**⚡ M4 Max hardware-accelerated NautilusTrader with 50x+ performance improvements**
- `/api/v1/nautilus/engine/start` - Start M4 Max optimized NautilusTrader with hardware acceleration
  - **New Features**: Metal Performance Shaders, Neural Engine integration, CPU affinity, unified memory
- `/api/v1/nautilus/engine/stop` - Stop engine with M4 Max hardware state preservation
- `/api/v1/nautilus/engine/restart` - Restart with M4 Max optimization configuration
- `/api/v1/nautilus/engine/status` - Engine status with M4 Max hardware metrics and performance data
- `/api/v1/nautilus/engine/config` - Update engine configuration with M4 Max optimization parameters
- `/api/v1/nautilus/engine/logs` - Real-time logs with M4 Max hardware metrics integration
- `/api/v1/nautilus/engine/health` - Engine health with M4 Max thermal and performance monitoring
- `/api/v1/nautilus/engine/emergency-stop` - Emergency stop with M4 Max resource cleanup
- `/api/v1/nautilus/engine/backtest` - M4 Max accelerated backtesting with Neural Engine

### M4 Max Engine Configuration Parameters
```json
{
  "config": {
    "engine_type": "live",
    "instance_id": "m4max-001",
    "hardware_acceleration": true,
    "metal_performance_shaders": true,
    "neural_engine": true,
    "cpu_affinity": "performance_cores",
    "max_memory": "12g",
    "max_cpu": "4.0",
    "unified_memory_optimization": true,
    "thermal_management": true
  }
}
```

## NautilusTrader Engine Management (Legacy Sprint 2)
**🚨 CRITICAL: This is now REAL NautilusTrader integration, NOT mocks**
- `/api/v1/nautilus/engine/start` - Start real NautilusTrader container with live configuration
- `/api/v1/nautilus/engine/stop` - Stop NautilusTrader engine (graceful or force)
- `/api/v1/nautilus/engine/restart` - Restart engine with current configuration
- `/api/v1/nautilus/engine/status` - **FIXED**: Returns flattened structures for single container
- `/api/v1/nautilus/engine/config` - Update engine configuration
- `/api/v1/nautilus/engine/logs` - Get real-time engine logs from container
- `/api/v1/nautilus/engine/health` - Engine health check from running container
- `/api/v1/nautilus/engine/emergency-stop` - Emergency force stop
- `/api/v1/nautilus/engine/backtest` - Start backtest in dedicated container

## Sprint 3: Advanced Analytics & Performance
**📊 NEW: Real-time analytics and performance monitoring**
- `/api/v1/analytics/performance/{portfolio_id}` - Real-time P&L and performance metrics
- `/api/v1/analytics/risk/{portfolio_id}` - VaR calculations and risk analytics
- `/api/v1/analytics/strategy/{strategy_id}` - Strategy performance analysis
- `/api/v1/analytics/execution/{execution_id}` - Trade execution quality analysis
- `/api/v1/analytics/aggregate` - Data aggregation and compression

## Sprint 3: Dynamic Risk Management
**⚠️ NEW: Advanced risk management with ML-based predictions**
- `/api/v1/risk/limits` - Dynamic risk limit CRUD operations
- `/api/v1/risk/limits/{limit_id}/check` - Real-time limit validation
- `/api/v1/risk/breaches` - Breach detection and management
- `/api/v1/risk/monitor/start` - Start real-time risk monitoring
- `/api/v1/risk/monitor/stop` - Stop risk monitoring
- `/api/v1/risk/reports/{report_type}` - Multi-format risk reporting
- `/api/v1/risk/alerts` - Risk alert management

## Sprint 3: Strategy Deployment Framework
**🚀 NEW: Automated strategy deployment with CI/CD pipelines**
- `/api/v1/strategies/deploy` - Deploy strategy with approval workflows
- `/api/v1/strategies/test/{strategy_id}` - Automated strategy testing
- `/api/v1/strategies/versions/{strategy_id}` - Version control operations
- `/api/v1/strategies/rollback/{deployment_id}` - Automated rollback procedures
- `/api/v1/strategies/pipeline/{pipeline_id}/status` - Deployment pipeline monitoring

## M4 Max Hardware Monitoring & Optimization ⚡ **PRODUCTION READY**
**🏆 M4 Max hardware acceleration with real-time monitoring and optimization**

### M4 Max Hardware Metrics
- `/api/v1/monitoring/m4max/hardware/metrics` - Real-time M4 Max hardware metrics (CPU P-cores/E-cores, unified memory, GPU, Neural Engine, thermal state)
- `/api/v1/monitoring/m4max/hardware/history` - M4 Max hardware metrics history with configurable time window
- `/api/v1/monitoring/containers/metrics` - Container performance metrics with M4 Max resource allocation
- `/api/v1/monitoring/trading/metrics` - Trading performance metrics with hardware acceleration insights
- `/api/v1/monitoring/dashboard/summary` - Comprehensive production dashboard with M4 Max status
- `/api/v1/monitoring/system/health` - Overall system health including M4 Max hardware status
- `/api/v1/monitoring/alerts` - M4 Max performance alerts and notifications
- `/api/v1/monitoring/performance/optimizations` - Performance optimization recommendations
- `/api/v1/monitoring/status` - M4 Max monitoring services status and configuration

### CPU Optimization & Performance
- `/api/v1/optimization/health` - CPU optimization system health with M4 Max core utilization
- `/api/v1/optimization/stats` - Comprehensive system statistics including M4 Max performance
- `/api/v1/optimization/core-utilization` - Per-core CPU utilization (P-cores vs E-cores)
- `/api/v1/optimization/register-process` - Register process for M4 Max optimization
- `/api/v1/optimization/classify-workload` - Classify workload for optimal M4 Max resource allocation
- `/api/v1/optimization/optimization-mode` - Get/set optimization mode (performance/efficiency/balanced)
- `/api/v1/optimization/rebalance-workloads` - Trigger M4 Max workload rebalancing
- `/api/v1/optimization/system-info` - M4 Max system capabilities and configuration

### Container Optimization
- `/api/v1/optimization/containers/stats` - M4 Max container optimization statistics
- `/api/v1/optimization/containers/performance-analysis` - Detailed container performance analysis
- `/api/v1/optimization/containers/optimize` - Force M4 Max container optimization
- `/api/v1/optimization/containers/{container_name}/priority` - Update container priority for M4 Max resource allocation

### Performance Monitoring & Analytics
- `/api/v1/optimization/latency-stats` - Real-time latency statistics with M4 Max acceleration impact
- `/api/v1/optimization/process-stats` - Process management statistics with CPU affinity
- `/api/v1/optimization/gcd-stats` - Grand Central Dispatch statistics for M4 Max scheduling
- `/api/v1/optimization/alerts` - Active performance alerts with M4 Max metrics
- `/api/v1/optimization/export-performance-data` - Export M4 Max performance data for analysis

## Sprint 3: System Monitoring & Health (Legacy)
**📈 Comprehensive system monitoring with Prometheus/Grafana**
- `/api/v1/system/health` - System health across all Sprint 3 components
- `/api/v1/system/metrics` - Performance metrics collection
- `/api/v1/system/alerts` - System alert management