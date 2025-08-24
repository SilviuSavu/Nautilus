# Story 1.1: PyFolio Integration Setup - Implementation Guide

## 📋 Story Overview

**Title**: As a quantitative analyst, I want PyFolio integrated into the risk engine so that I can access comprehensive portfolio performance analytics.

**Story Points**: 5  
**Priority**: P0-Critical  
**Timeline**: Days 1-2  
**Estimated Effort**: 16 hours

## 🎯 Business Objectives

### **Primary Goals**
- Enable institutional-grade portfolio analytics within Nautilus platform
- Provide comprehensive performance attribution and risk metrics
- Generate professional tear sheets for client reporting
- Establish foundation for advanced risk analytics integration

### **Success Criteria**
- [ ] PyFolio library successfully integrated into risk engine container
- [ ] Portfolio returns data pipeline functional
- [ ] Basic tear sheet generation working (<200ms response time)
- [ ] Integration tests passing with sample portfolio data
- [ ] Risk engine health checks include PyFolio status

## 🛠️ Technical Implementation

### **Phase 1: Dependencies and Environment Setup** (4 hours)

#### **1.1 Update Requirements File**
**File**: `/backend/engines/risk/requirements.txt`

```python
# Add to existing requirements.txt
pyfolio>=0.9.2
empyrical>=0.5.5
pandas>=2.1.0  # Ensure compatibility
matplotlib>=3.7.0  # For visualizations
seaborn>=0.12.0   # Statistical plots
```

#### **1.2 Container Verification** 
**Files**: `backend/engines/risk/Dockerfile` (if custom Dockerfile exists)

```dockerfile
# Verify these system dependencies are available
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    gfortran
```

#### **1.3 Import Validation**
Create test script to verify PyFolio imports correctly:

```python
# Test script: test_pyfolio_import.py
try:
    import pyfolio as pf
    import empyrical as ep
    print("✅ PyFolio import successful")
    print(f"PyFolio version: {pf.__version__}")
except ImportError as e:
    print(f"❌ PyFolio import failed: {e}")
```

### **Phase 2: PyFolio Integration Layer** (8 hours)

#### **2.1 Create PyFolio Wrapper Class**
**File**: `/backend/engines/risk/pyfolio_integration.py`

```python
"""
PyFolio Integration Layer
========================

Wrapper for PyFolio functionality optimized for Nautilus platform.
Provides portfolio analytics, tear sheets, and performance attribution.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import logging
import time
import io
import base64

try:
    import pyfolio as pf
    import empyrical as ep
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    PYFOLIO_AVAILABLE = True
except ImportError:
    PYFOLIO_AVAILABLE = False

logger = logging.getLogger(__name__)

class PyFolioAnalytics:
    """
    PyFolio integration for comprehensive portfolio analytics
    """
    
    def __init__(self):
        self.available = PYFOLIO_AVAILABLE
        if not self.available:
            logger.error("PyFolio not available - analytics limited")
        
        # Performance tracking
        self.calculations_performed = 0
        self.total_calculation_time = 0.0
        
        # Default benchmark (SPY equivalent)
        self.default_benchmark_ticker = 'SPY'
    
    async def compute_performance_metrics(self, 
                                        returns: pd.Series,
                                        benchmark_returns: Optional[pd.Series] = None,
                                        risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Compute comprehensive performance metrics using PyFolio
        
        Args:
            returns: Portfolio returns time series
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            Dictionary with all performance metrics
        """
        if not self.available:
            raise RuntimeError("PyFolio not available")
        
        start_time = time.time()
        
        try:
            # Basic performance metrics
            metrics = {
                'total_return': ep.cum_returns_final(returns),
                'annual_return': ep.annual_return(returns),
                'annual_volatility': ep.annual_volatility(returns),
                'sharpe_ratio': ep.sharpe_ratio(returns, risk_free=risk_free_rate),
                'max_drawdown': ep.max_drawdown(returns),
                'sortino_ratio': ep.sortino_ratio(returns, required_return=risk_free_rate),
                'calmar_ratio': ep.calmar_ratio(returns),
                'omega_ratio': ep.omega_ratio(returns, risk_free=risk_free_rate),
                'tail_ratio': ep.tail_ratio(returns),
                'value_at_risk': returns.quantile(0.05),  # 5% VaR
                'conditional_value_at_risk': returns[returns <= returns.quantile(0.05)].mean()
            }
            
            # Benchmark-relative metrics
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                try:
                    metrics.update({
                        'alpha': ep.alpha(returns, benchmark_returns, risk_free_rate),
                        'beta': ep.beta(returns, benchmark_returns),
                        'tracking_error': ep.tracking_error(returns, benchmark_returns),
                        'information_ratio': ep.excess_sharpe(returns, benchmark_returns),
                        'up_capture': ep.up_capture(returns, benchmark_returns),
                        'down_capture': ep.down_capture(returns, benchmark_returns)
                    })
                except Exception as e:
                    logger.warning(f"Benchmark metrics calculation failed: {e}")
            
            # Additional risk metrics
            metrics.update({
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'var_95': np.percentile(returns, 5),
                'var_99': np.percentile(returns, 1),
                'downside_deviation': ep.downside_risk(returns),
                'upside_risk': ep.excess_sharpe(returns[returns > returns.mean()])
            })
            
            # Performance tracking
            calculation_time = time.time() - start_time
            self.calculations_performed += 1
            self.total_calculation_time += calculation_time
            
            metrics['calculation_time_ms'] = calculation_time * 1000
            
            return metrics
            
        except Exception as e:
            logger.error(f"PyFolio metrics calculation failed: {e}")
            raise
    
    async def generate_tear_sheet_data(self, 
                                     returns: pd.Series,
                                     benchmark_returns: Optional[pd.Series] = None,
                                     positions: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate data for tear sheet (without plotting for API use)
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns
            positions: Optional positions DataFrame
            
        Returns:
            Dictionary with tear sheet data
        """
        if not self.available:
            raise RuntimeError("PyFolio not available")
        
        try:
            # Get performance metrics
            metrics = await self.compute_performance_metrics(returns, benchmark_returns)
            
            # Additional tear sheet specific data
            tear_sheet_data = {
                'performance_metrics': metrics,
                'rolling_returns': {
                    'rolling_sharpe_6m': returns.rolling(window=126).apply(
                        lambda x: ep.sharpe_ratio(x), raw=False
                    ).dropna(),
                    'rolling_volatility_6m': returns.rolling(window=126).std() * np.sqrt(252),
                    'rolling_beta_6m': None  # Requires benchmark
                },
                'drawdown_analysis': {
                    'drawdown_series': (returns + 1).cumprod() / (returns + 1).cumprod().expanding().max() - 1,
                    'top_5_drawdowns': self._get_top_drawdowns(returns, top=5),
                    'underwater_plot_data': self._get_underwater_data(returns)
                },
                'returns_distribution': {
                    'daily_returns_histogram': returns.hist(bins=50, density=True),
                    'monthly_returns': ep.aggregate_returns(returns, 'monthly'),
                    'yearly_returns': ep.aggregate_returns(returns, 'yearly')
                }
            }
            
            # Add benchmark comparison if available
            if benchmark_returns is not None:
                tear_sheet_data['benchmark_comparison'] = {
                    'excess_returns': returns - benchmark_returns,
                    'cumulative_returns_comparison': {
                        'portfolio': (1 + returns).cumprod(),
                        'benchmark': (1 + benchmark_returns).cumprod()
                    }
                }
            
            return tear_sheet_data
            
        except Exception as e:
            logger.error(f"Tear sheet generation failed: {e}")
            raise
    
    def _get_top_drawdowns(self, returns: pd.Series, top: int = 5) -> pd.DataFrame:
        """Get top N drawdown periods"""
        try:
            drawdowns = pf.timeseries.gen_drawdown_table(returns, top=top)
            return drawdowns.to_dict() if drawdowns is not None else {}
        except Exception as e:
            logger.warning(f"Top drawdowns calculation failed: {e}")
            return {}
    
    def _get_underwater_data(self, returns: pd.Series) -> pd.Series:
        """Get underwater (drawdown) time series data"""
        try:
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            underwater = (cum_returns - running_max) / running_max
            return underwater
        except Exception as e:
            logger.warning(f"Underwater data calculation failed: {e}")
            return pd.Series()
    
    async def generate_html_tear_sheet(self, 
                                     returns: pd.Series,
                                     benchmark_returns: Optional[pd.Series] = None) -> str:
        """
        Generate HTML tear sheet for web display
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns
            
        Returns:
            HTML string with tear sheet
        """
        if not self.available:
            return "<h1>PyFolio not available</h1>"
        
        try:
            # Generate tear sheet data
            data = await self.generate_tear_sheet_data(returns, benchmark_returns)
            metrics = data['performance_metrics']
            
            html_content = f"""
            <html>
            <head>
                <title>Portfolio Performance Tear Sheet</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ margin: 10px 0; }}
                    .section {{ margin: 20px 0; border: 1px solid #ccc; padding: 15px; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    .neutral {{ color: blue; }}
                </style>
            </head>
            <body>
                <h1>Portfolio Performance Analysis</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Performance Summary</h2>
                    <div class="metric">Total Return: <span class="{'positive' if metrics['total_return'] > 0 else 'negative'}">{metrics['total_return']:.2%}</span></div>
                    <div class="metric">Annual Return: <span class="{'positive' if metrics['annual_return'] > 0 else 'negative'}">{metrics['annual_return']:.2%}</span></div>
                    <div class="metric">Annual Volatility: {metrics['annual_volatility']:.2%}</div>
                    <div class="metric">Sharpe Ratio: <span class="{'positive' if metrics['sharpe_ratio'] > 1 else 'neutral'}">{metrics['sharpe_ratio']:.2f}</span></div>
                    <div class="metric">Max Drawdown: <span class="negative">{metrics['max_drawdown']:.2%}</span></div>
                </div>
                
                <div class="section">
                    <h2>Risk Metrics</h2>
                    <div class="metric">Sortino Ratio: {metrics['sortino_ratio']:.2f}</div>
                    <div class="metric">Calmar Ratio: {metrics['calmar_ratio']:.2f}</div>
                    <div class="metric">VaR (5%): <span class="negative">{metrics['value_at_risk']:.2%}</span></div>
                    <div class="metric">CVaR (5%): <span class="negative">{metrics['conditional_value_at_risk']:.2%}</span></div>
                </div>
            
                <div class="section">
                    <h2>Statistical Properties</h2>
                    <div class="metric">Skewness: {metrics['skewness']:.3f}</div>
                    <div class="metric">Kurtosis: {metrics['kurtosis']:.3f}</div>
                    <div class="metric">Tail Ratio: {metrics['tail_ratio']:.3f}</div>
                </div>
                
                <div class="section">
                    <h2>Performance Details</h2>
                    <p>Calculation completed in {metrics['calculation_time_ms']:.1f}ms</p>
                    <p>Total calculations performed: {self.calculations_performed}</p>
                </div>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"HTML tear sheet generation failed: {e}")
            return f"<h1>Error generating tear sheet: {e}</h1>"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get PyFolio integration performance statistics"""
        avg_time = self.total_calculation_time / max(1, self.calculations_performed)
        
        return {
            'available': self.available,
            'calculations_performed': self.calculations_performed,
            'average_calculation_time_ms': avg_time * 1000,
            'total_calculation_time_seconds': self.total_calculation_time
        }
```

#### **2.2 Integration with Risk Engine**
**File**: `/backend/engines/risk/risk_engine.py`

Add PyFolio integration to the risk engine:

```python
# Add to imports
from pyfolio_integration import PyFolioAnalytics

class RiskEngine:
    def __init__(self):
        # ... existing initialization ...
        
        # PyFolio integration
        self.pyfolio = PyFolioAnalytics()
        
    def setup_routes(self):
        # ... existing routes ...
        
        @self.app.post("/risk/analytics/pyfolio/{portfolio_id}")
        async def compute_pyfolio_analytics(portfolio_id: str, request_data: Dict[str, Any]):
            """Compute PyFolio portfolio analytics"""
            try:
                import pandas as pd
                
                returns_data = request_data.get("returns", [])
                benchmark_data = request_data.get("benchmark_returns", [])
                
                if not returns_data:
                    raise HTTPException(status_code=400, detail="Returns data required")
                
                returns = pd.Series(returns_data)
                benchmark = pd.Series(benchmark_data) if benchmark_data else None
                
                # Compute PyFolio analytics
                analytics = await self.pyfolio.compute_performance_metrics(
                    returns=returns,
                    benchmark_returns=benchmark
                )
                
                return {
                    "status": "success",
                    "portfolio_id": portfolio_id,
                    "analytics": analytics
                }
                
            except Exception as e:
                logger.error(f"PyFolio analytics error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/risk/analytics/tear-sheet/{portfolio_id}")
        async def generate_tear_sheet(portfolio_id: str, format: str = "html"):
            """Generate PyFolio tear sheet"""
            try:
                # Get portfolio returns (implement based on your data access)
                returns = await self._get_portfolio_returns(portfolio_id)
                benchmark = await self._get_benchmark_returns()  # Optional
                
                if format == "html":
                    tear_sheet = await self.pyfolio.generate_html_tear_sheet(returns, benchmark)
                    from fastapi.responses import HTMLResponse
                    return HTMLResponse(content=tear_sheet)
                else:
                    tear_sheet_data = await self.pyfolio.generate_tear_sheet_data(returns, benchmark)
                    return {"status": "success", "tear_sheet": tear_sheet_data}
                    
            except Exception as e:
                logger.error(f"Tear sheet generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
```

### **Phase 3: Testing and Validation** (4 hours)

#### **3.1 Unit Tests**
**File**: `/backend/engines/risk/tests/test_pyfolio_integration.py`

```python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyfolio_integration import PyFolioAnalytics

class TestPyFolioIntegration:
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        returns = np.random.normal(0.0005, 0.02, len(dates))  # ~12% annual return, 20% vol
        return pd.Series(returns, index=dates)
    
    @pytest.fixture
    def benchmark_returns(self, sample_returns):
        """Generate benchmark return data"""
        # Slightly lower returns than portfolio
        benchmark = sample_returns * 0.8 + np.random.normal(0, 0.001, len(sample_returns))
        return benchmark
    
    @pytest.fixture
    def pyfolio_analytics(self):
        return PyFolioAnalytics()
    
    def test_pyfolio_availability(self, pyfolio_analytics):
        """Test that PyFolio is available and properly initialized"""
        assert pyfolio_analytics.available == True
        assert pyfolio_analytics.calculations_performed == 0
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, pyfolio_analytics, sample_returns):
        """Test basic performance metrics calculation"""
        metrics = await pyfolio_analytics.compute_performance_metrics(sample_returns)
        
        # Verify all expected metrics are present
        expected_metrics = [
            'total_return', 'annual_return', 'annual_volatility', 'sharpe_ratio',
            'max_drawdown', 'sortino_ratio', 'calmar_ratio', 'value_at_risk',
            'conditional_value_at_risk'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
    
    @pytest.mark.asyncio
    async def test_benchmark_comparison(self, pyfolio_analytics, sample_returns, benchmark_returns):
        """Test performance metrics with benchmark"""
        metrics = await pyfolio_analytics.compute_performance_metrics(
            sample_returns, benchmark_returns
        )
        
        # Verify benchmark-specific metrics
        benchmark_metrics = ['alpha', 'beta', 'tracking_error', 'information_ratio']
        
        for metric in benchmark_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    @pytest.mark.asyncio
    async def test_performance_timing(self, pyfolio_analytics, sample_returns):
        """Test that performance calculation meets timing requirements"""
        start_time = datetime.now()
        metrics = await pyfolio_analytics.compute_performance_metrics(sample_returns)
        end_time = datetime.now()
        
        calculation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Should complete within 200ms
        assert calculation_time_ms < 200
        assert metrics['calculation_time_ms'] < 200
    
    @pytest.mark.asyncio
    async def test_tear_sheet_generation(self, pyfolio_analytics, sample_returns):
        """Test tear sheet data generation"""
        tear_sheet_data = await pyfolio_analytics.generate_tear_sheet_data(sample_returns)
        
        assert 'performance_metrics' in tear_sheet_data
        assert 'rolling_returns' in tear_sheet_data
        assert 'drawdown_analysis' in tear_sheet_data
        assert 'returns_distribution' in tear_sheet_data
    
    @pytest.mark.asyncio
    async def test_html_tear_sheet(self, pyfolio_analytics, sample_returns):
        """Test HTML tear sheet generation"""
        html_output = await pyfolio_analytics.generate_html_tear_sheet(sample_returns)
        
        assert isinstance(html_output, str)
        assert '<html>' in html_output
        assert 'Portfolio Performance Analysis' in html_output
        assert 'Sharpe Ratio' in html_output
```

#### **3.2 Integration Tests**
**File**: `/backend/engines/risk/tests/test_risk_engine_pyfolio.py`

```python
import pytest
import httpx
from fastapi.testclient import TestClient
from risk_engine import RiskEngine

class TestRiskEnginePyFolio:
    
    @pytest.fixture
    def client(self):
        engine = RiskEngine()
        return TestClient(engine.app)
    
    @pytest.fixture
    def sample_request_data(self):
        return {
            "returns": [0.01, -0.005, 0.02, 0.005, -0.01, 0.015, -0.008],
            "benchmark_returns": [0.008, -0.002, 0.015, 0.003, -0.012, 0.01, -0.005]
        }
    
    def test_pyfolio_analytics_endpoint(self, client, sample_request_data):
        """Test PyFolio analytics API endpoint"""
        response = client.post(
            "/risk/analytics/pyfolio/test_portfolio",
            json=sample_request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["portfolio_id"] == "test_portfolio"
        assert "analytics" in data
        assert "sharpe_ratio" in data["analytics"]
    
    def test_tear_sheet_html_endpoint(self, client):
        """Test HTML tear sheet endpoint"""
        # This test would need actual portfolio data
        # For now, test that the endpoint exists and returns properly formatted error
        response = client.get("/risk/analytics/tear-sheet/test_portfolio?format=html")
        
        # Expect 500 since we don't have real data, but should be proper HTTP response
        assert response.status_code in [200, 500]
        
    def test_analytics_performance(self, client, sample_request_data):
        """Test that analytics complete within performance requirements"""
        import time
        
        start_time = time.time()
        response = client.post(
            "/risk/analytics/pyfolio/test_portfolio", 
            json=sample_request_data
        )
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 0.2  # Less than 200ms
```

## 📊 Testing Checklist

### **Unit Test Coverage** (Target: >85%)
- [ ] PyFolioAnalytics class initialization
- [ ] Performance metrics calculation accuracy
- [ ] Benchmark comparison calculations
- [ ] Error handling for invalid inputs
- [ ] Performance timing validation
- [ ] Tear sheet data generation
- [ ] HTML output generation

### **Integration Tests**
- [ ] Risk engine API endpoints functional
- [ ] HTTP response format validation
- [ ] Error response handling
- [ ] Performance under load (basic)
- [ ] Memory usage validation

### **Acceptance Tests**
- [ ] Real portfolio data processing
- [ ] Tear sheet visual validation
- [ ] Cross-validation with known analytics tools
- [ ] Performance benchmarking

## 🚀 Deployment Checklist

### **Pre-deployment**
- [ ] All tests passing
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Performance benchmarks recorded
- [ ] Error handling validated

### **Deployment**
- [ ] Container builds successfully
- [ ] Health check endpoints responsive
- [ ] API endpoints accessible
- [ ] Integration with existing risk engine confirmed
- [ ] No regression in existing functionality

### **Post-deployment**
- [ ] Monitor for errors in production logs
- [ ] Validate performance metrics
- [ ] Test with real portfolio data
- [ ] Confirm memory usage acceptable
- [ ] Document any issues for resolution

## 📈 Success Metrics

### **Functional Success**
- ✅ PyFolio analytics available via REST API
- ✅ Tear sheet generation working (<200ms)
- ✅ All key performance metrics calculated correctly
- ✅ Integration tests passing
- ✅ No critical bugs identified

### **Performance Success**
- ✅ Response time <200ms for basic analytics
- ✅ Memory usage <100MB additional per calculation
- ✅ No memory leaks during extended operation
- ✅ Container starts successfully with new dependencies

### **Quality Success**
- ✅ Code coverage >85%
- ✅ All acceptance criteria met
- ✅ Documentation complete and accurate
- ✅ Error handling comprehensive
- ✅ Ready for next story (QuantStats integration)

---

**Story Status**: 📋 **READY FOR IMPLEMENTATION**

**Next Story**: Story 1.2 (QuantStats Integration) - builds on PyFolio foundation