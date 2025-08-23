# Sprint 3 Developer Guide

## Overview

This comprehensive developer guide provides detailed instructions for extending Sprint 3 functionality, including adding new components, creating custom analytics, implementing risk strategies, and integrating additional data sources.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Environment Setup](#development-environment-setup)
3. [Adding New Components](#adding-new-components)
4. [Custom Analytics Development](#custom-analytics-development)
5. [Risk Strategy Extension](#risk-strategy-extension)
6. [WebSocket Service Extension](#websocket-service-extension)
7. [Data Source Integration](#data-source-integration)
8. [API Endpoint Development](#api-endpoint-development)
9. [Testing Framework](#testing-framework)
10. [Performance Optimization](#performance-optimization)

---

## Architecture Overview

### Sprint 3 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  React Components │ WebSocket Hooks │ Real-time Analytics      │
│  Risk Dashboard   │ Strategy UI     │ Monitoring Suite         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Routes   │ Authentication  │ Rate Limiting            │
│  Input Validation │ Error Handling  │ Request Correlation      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Service Layer                               │
├─────────────────────────────────────────────────────────────────┤
│ Analytics Service │ Risk Service    │ Strategy Service         │
│ WebSocket Manager │ Limit Engine    │ Deployment Manager       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer                                  │
├─────────────────────────────────────────────────────────────────┤
│ PostgreSQL/TimescaleDB │ Redis Cache │ Message Bus             │
│ Real-time Streams      │ Time Series │ Event Processing        │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Modularity**: Each component is self-contained with clear interfaces
2. **Scalability**: Horizontal scaling through microservices pattern
3. **Real-time**: WebSocket-first architecture for live updates
4. **Extensibility**: Plugin-based system for easy feature addition
5. **Observability**: Comprehensive logging and monitoring
6. **Security**: Authentication, authorization, and input validation

---

## Development Environment Setup

### Prerequisites Installation

```bash
# Install development dependencies
pip install -r requirements-dev.txt
npm install --dev

# Install pre-commit hooks
pre-commit install

# Setup environment variables
cp .env.example .env.development
```

### Development Database Setup

```sql
-- Create development database with Sprint 3 extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create hypertables for time-series data
SELECT create_hypertable('analytics_metrics', 'timestamp');
SELECT create_hypertable('risk_events', 'timestamp');
SELECT create_hypertable('websocket_metrics', 'timestamp');

-- Create indexes for performance
CREATE INDEX idx_analytics_portfolio_time ON analytics_metrics (portfolio_id, timestamp DESC);
CREATE INDEX idx_risk_events_type_time ON risk_events (event_type, timestamp DESC);
```

### Code Quality Tools

```bash
# Backend code quality
black backend/
isort backend/
flake8 backend/
mypy backend/

# Frontend code quality
npm run lint
npm run type-check
npm run format
```

---

## Adding New Components

### Frontend Component Development

#### 1. Component Structure Template

```tsx
// src/components/NewFeature/NewComponent.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { useWebSocketManager } from '@/hooks';
import { NewComponentProps, ComponentState } from './types';

interface NewComponentProps {
  portfolioId: string;
  enableRealTime?: boolean;
  onDataUpdate?: (data: ComponentData) => void;
}

export const NewComponent: React.FC<NewComponentProps> = ({
  portfolioId,
  enableRealTime = true,
  onDataUpdate
}) => {
  const [state, setState] = useState<ComponentState>({
    loading: true,
    data: null,
    error: null
  });

  const { subscribe, unsubscribe } = useWebSocketManager();

  // Real-time data subscription
  useEffect(() => {
    if (!enableRealTime) return;

    const subscription = subscribe(
      'new_feature.updates',
      { portfolio_id: portfolioId },
      (data) => {
        setState(prev => ({ ...prev, data, loading: false }));
        onDataUpdate?.(data);
      }
    );

    return () => unsubscribe(subscription.id);
  }, [portfolioId, enableRealTime]);

  // API data fetching
  const fetchData = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      const response = await fetch(`/api/v1/sprint3/new-feature/${portfolioId}`);
      const data = await response.json();
      
      setState({ data, loading: false, error: null });
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        loading: false, 
        error: error.message 
      }));
    }
  }, [portfolioId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  if (state.loading) {
    return <div className="loading">Loading new component...</div>;
  }

  if (state.error) {
    return <div className="error">Error: {state.error}</div>;
  }

  return (
    <div className="new-component">
      <h3>New Component</h3>
      <pre>{JSON.stringify(state.data, null, 2)}</pre>
    </div>
  );
};

export default NewComponent;
```

#### 2. Component Types Definition

```typescript
// src/components/NewFeature/types.ts
export interface ComponentData {
  id: string;
  portfolio_id: string;
  metrics: Record<string, number>;
  timestamp: string;
}

export interface ComponentState {
  loading: boolean;
  data: ComponentData | null;
  error: string | null;
}

export interface NewComponentProps {
  portfolioId: string;
  enableRealTime?: boolean;
  refreshInterval?: number;
  onDataUpdate?: (data: ComponentData) => void;
  onError?: (error: Error) => void;
}
```

#### 3. Custom Hook Development

```typescript
// src/hooks/newFeature/useNewFeature.ts
import { useState, useEffect, useCallback } from 'react';
import { ComponentData } from '@/components/NewFeature/types';

export const useNewFeature = (portfolioId: string) => {
  const [data, setData] = useState<ComponentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`/api/v1/sprint3/new-feature/${portfolioId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      setData(result.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [portfolioId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    data,
    loading,
    error,
    refetch: fetchData
  };
};
```

### Backend Service Development

#### 1. Service Class Template

```python
# backend/services/new_feature_service.py
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from .base_service import BaseService
from ..models.new_feature import NewFeatureModel
from ..schemas.new_feature import NewFeatureSchema, NewFeatureCreate


class NewFeatureService(BaseService):
    """New feature service with real-time capabilities."""
    
    def __init__(self, db: AsyncSession, redis: Redis):
        super().__init__(db, redis)
        self.model = NewFeatureModel
        self.schema = NewFeatureSchema
    
    async def create_feature(
        self, 
        portfolio_id: str, 
        data: NewFeatureCreate
    ) -> NewFeatureSchema:
        """Create a new feature instance."""
        try:
            # Validate input data
            validated_data = self.validate_input(data)
            
            # Create database record
            db_feature = self.model(
                portfolio_id=portfolio_id,
                **validated_data.dict(),
                created_at=datetime.utcnow()
            )
            
            self.db.add(db_feature)
            await self.db.commit()
            await self.db.refresh(db_feature)
            
            # Cache the result
            await self.cache_feature(db_feature)
            
            # Broadcast update
            await self.broadcast_update(
                'new_feature.created',
                self.schema.from_orm(db_feature).dict()
            )
            
            return self.schema.from_orm(db_feature)
            
        except Exception as e:
            await self.db.rollback()
            raise self.handle_error(e, "create_feature")
    
    async def get_features(
        self,
        portfolio_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[NewFeatureSchema]:
        """Get features with caching and pagination."""
        cache_key = f"features:{portfolio_id}:{limit}:{offset}"
        
        # Try cache first
        cached_data = await self.get_cached(cache_key)
        if cached_data:
            return [self.schema(**item) for item in cached_data]
        
        # Query database
        query = (
            select(self.model)
            .where(self.model.portfolio_id == portfolio_id)
            .order_by(self.model.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        result = await self.db.execute(query)
        features = result.scalars().all()
        
        # Convert to schemas
        feature_schemas = [self.schema.from_orm(f) for f in features]
        
        # Cache results
        await self.set_cached(
            cache_key,
            [f.dict() for f in feature_schemas],
            ttl=300  # 5 minutes
        )
        
        return feature_schemas
    
    async def update_real_time_metrics(
        self,
        portfolio_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """Update real-time metrics and broadcast."""
        try:
            # Update database
            update_stmt = (
                update(self.model)
                .where(self.model.portfolio_id == portfolio_id)
                .values(
                    metrics=metrics,
                    updated_at=datetime.utcnow()
                )
            )
            
            await self.db.execute(update_stmt)
            await self.db.commit()
            
            # Broadcast real-time update
            await self.broadcast_update(
                'new_feature.metrics_updated',
                {
                    'portfolio_id': portfolio_id,
                    'metrics': metrics,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            await self.db.rollback()
            raise self.handle_error(e, "update_real_time_metrics")
    
    async def cache_feature(self, feature: NewFeatureModel) -> None:
        """Cache feature data."""
        cache_key = f"feature:{feature.id}"
        feature_data = self.schema.from_orm(feature).dict()
        await self.set_cached(cache_key, feature_data, ttl=3600)
    
    async def broadcast_update(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast update via Redis pub/sub."""
        message = {
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.redis.publish('new_feature_updates', json.dumps(message))
```

#### 2. API Route Implementation

```python
# backend/routes/new_feature_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from ..dependencies import get_db, get_redis, get_current_user
from ..services.new_feature_service import NewFeatureService
from ..schemas.new_feature import NewFeatureSchema, NewFeatureCreate

router = APIRouter(prefix="/api/v1/sprint3/new-feature", tags=["new-feature"])


@router.post("/{portfolio_id}", response_model=NewFeatureSchema)
async def create_feature(
    portfolio_id: str,
    feature_data: NewFeatureCreate,
    db=Depends(get_db),
    redis=Depends(get_redis),
    current_user=Depends(get_current_user)
):
    """Create a new feature."""
    service = NewFeatureService(db, redis)
    return await service.create_feature(portfolio_id, feature_data)


@router.get("/{portfolio_id}", response_model=List[NewFeatureSchema])
async def get_features(
    portfolio_id: str,
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    db=Depends(get_db),
    redis=Depends(get_redis)
):
    """Get features for portfolio."""
    service = NewFeatureService(db, redis)
    return await service.get_features(portfolio_id, limit, offset)


@router.put("/{portfolio_id}/metrics")
async def update_metrics(
    portfolio_id: str,
    metrics: dict,
    db=Depends(get_db),
    redis=Depends(get_redis),
    current_user=Depends(get_current_user)
):
    """Update real-time metrics."""
    service = NewFeatureService(db, redis)
    await service.update_real_time_metrics(portfolio_id, metrics)
    return {"status": "success"}


@router.get("/{portfolio_id}/health")
async def health_check(
    portfolio_id: str,
    db=Depends(get_db),
    redis=Depends(get_redis)
):
    """Health check for new feature service."""
    service = NewFeatureService(db, redis)
    
    try:
        # Test database connection
        await service.get_features(portfolio_id, limit=1)
        
        # Test Redis connection
        await service.redis.ping()
        
        return {
            "status": "healthy",
            "database": "connected",
            "redis": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
```

---

## Custom Analytics Development

### Creating Custom Analytics Modules

#### 1. Analytics Base Class

```python
# backend/analytics/base_analyzer.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class BaseAnalyzer(ABC):
    """Base class for all analytics modules."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.created_at = datetime.utcnow()
    
    @abstractmethod
    async def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Main calculation method - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get analyzer metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "description": self.__doc__ or "No description available"
        }
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data before analysis."""
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Sort by timestamp
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(method='ffill')
        
        return data
```

#### 2. Custom Performance Analytics

```python
# backend/analytics/custom_performance_analyzer.py
from .base_analyzer import BaseAnalyzer
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class CustomPerformanceAnalyzer(BaseAnalyzer):
    """Custom performance analytics with advanced metrics."""
    
    def __init__(self):
        super().__init__("CustomPerformanceAnalyzer", "1.0.0")
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    async def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.validate_input(data):
            raise ValueError("Invalid input data format")
        
        data = self.preprocess_data(data)
        
        returns = self.calculate_returns(data)
        
        metrics = {
            "basic_metrics": self.calculate_basic_metrics(returns),
            "risk_metrics": self.calculate_risk_metrics(returns),
            "advanced_metrics": self.calculate_advanced_metrics(returns),
            "drawdown_analysis": self.calculate_drawdown_analysis(data),
            "rolling_metrics": self.calculate_rolling_metrics(returns)
        }
        
        return {
            "analyzer": self.get_metadata(),
            "metrics": metrics,
            "data_period": {
                "start": data.index[0].isoformat(),
                "end": data.index[-1].isoformat(),
                "periods": len(data)
            }
        }
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns."""
        required_columns = ['portfolio_value']
        return all(col in data.columns for col in required_columns)
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        """Calculate returns from portfolio values."""
        return data['portfolio_value'].pct_change().dropna()
    
    def calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float((annualized_return - self.risk_free_rate) / volatility)
        }
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-specific metrics."""
        # Value at Risk
        var_95 = float(returns.quantile(0.05))
        var_99 = float(returns.quantile(0.01))
        
        # Expected Shortfall (Conditional VaR)
        es_95 = float(returns[returns <= var_95].mean())
        es_99 = float(returns[returns <= var_99].mean())
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        downside_deviation = float(negative_returns.std() * np.sqrt(252))
        
        return {
            "var_95": var_95,
            "var_99": var_99,
            "expected_shortfall_95": es_95,
            "expected_shortfall_99": es_99,
            "downside_deviation": downside_deviation,
            "sortino_ratio": float((returns.mean() * 252 - self.risk_free_rate) / downside_deviation)
        }
    
    def calculate_advanced_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate advanced performance metrics."""
        # Skewness and Kurtosis
        skewness = float(returns.skew())
        kurtosis = float(returns.kurtosis())
        
        # Calmar Ratio (requires drawdown calculation)
        max_drawdown = self.calculate_max_drawdown(returns)
        annualized_return = (1 + returns.mean()) ** 252 - 1
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "calmar_ratio": float(calmar_ratio),
            "omega_ratio": self.calculate_omega_ratio(returns)
        }
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        return float(drawdowns.min())
    
    def calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns < 0].sum())
        
        return float(positive_returns / negative_returns) if negative_returns != 0 else float('inf')
```

#### 3. Real-time Analytics Engine

```python
# backend/analytics/real_time_engine.py
import asyncio
from typing import Dict, List, Any, Callable
from datetime import datetime
import pandas as pd
from redis.asyncio import Redis

from .custom_performance_analyzer import CustomPerformanceAnalyzer


class RealTimeAnalyticsEngine:
    """Real-time analytics processing engine."""
    
    def __init__(self, redis: Redis):
        self.redis = redis
        self.analyzers: Dict[str, BaseAnalyzer] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
    
    def register_analyzer(self, analyzer: BaseAnalyzer) -> None:
        """Register a new analyzer."""
        self.analyzers[analyzer.name] = analyzer
    
    def subscribe(self, portfolio_id: str, callback: Callable) -> None:
        """Subscribe to analytics updates for a portfolio."""
        if portfolio_id not in self.subscribers:
            self.subscribers[portfolio_id] = []
        self.subscribers[portfolio_id].append(callback)
    
    async def start(self) -> None:
        """Start the real-time analytics engine."""
        self.running = True
        
        # Start processing tasks
        tasks = [
            asyncio.create_task(self.process_market_data()),
            asyncio.create_task(self.process_portfolio_updates()),
            asyncio.create_task(self.cleanup_old_data())
        ]
        
        await asyncio.gather(*tasks)
    
    async def process_market_data(self) -> None:
        """Process incoming market data."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe('market_data_updates')
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    await self.update_analytics(data)
                except Exception as e:
                    logger.error(f"Error processing market data: {e}")
    
    async def update_analytics(self, data: Dict[str, Any]) -> None:
        """Update analytics for affected portfolios."""
        portfolio_id = data.get('portfolio_id')
        if not portfolio_id:
            return
        
        # Get recent portfolio data
        portfolio_data = await self.get_portfolio_data(portfolio_id)
        
        if len(portfolio_data) < 10:  # Need minimum data points
            return
        
        # Run all registered analyzers
        results = {}
        for name, analyzer in self.analyzers.items():
            try:
                result = await analyzer.calculate(portfolio_data)
                results[name] = result
            except Exception as e:
                logger.error(f"Error in analyzer {name}: {e}")
        
        # Notify subscribers
        if portfolio_id in self.subscribers:
            for callback in self.subscribers[portfolio_id]:
                try:
                    await callback(portfolio_id, results)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
        
        # Cache results
        await self.cache_results(portfolio_id, results)
    
    async def get_portfolio_data(self, portfolio_id: str, days: int = 30) -> pd.DataFrame:
        """Get recent portfolio data from database."""
        # Implementation would query TimescaleDB for recent data
        # This is a placeholder
        pass
    
    async def cache_results(self, portfolio_id: str, results: Dict[str, Any]) -> None:
        """Cache analytics results."""
        cache_key = f"analytics:real_time:{portfolio_id}"
        await self.redis.setex(
            cache_key,
            300,  # 5 minutes TTL
            json.dumps(results, default=str)
        )
```

---

## Risk Strategy Extension

### Custom Risk Models

#### 1. Advanced VaR Calculator

```python
# backend/risk/advanced_var_calculator.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from typing import Dict, Any, List, Optional

class AdvancedVaRCalculator:
    """Advanced VaR calculation with multiple methodologies."""
    
    def __init__(self):
        self.confidence_levels = [0.95, 0.99, 0.995]
        self.methods = [
            'historical',
            'parametric',
            'monte_carlo',
            'extreme_value',
            'copula'
        ]
    
    async def calculate_var(
        self,
        returns: pd.DataFrame,
        method: str = 'all',
        confidence_level: float = 0.95,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate VaR using specified method(s)."""
        
        results = {}
        
        if method == 'all' or method == 'historical':
            results['historical'] = self.historical_var(returns, confidence_level)
        
        if method == 'all' or method == 'parametric':
            results['parametric'] = self.parametric_var(returns, confidence_level)
        
        if method == 'all' or method == 'monte_carlo':
            results['monte_carlo'] = await self.monte_carlo_var(
                returns, confidence_level, kwargs.get('simulations', 10000)
            )
        
        if method == 'all' or method == 'extreme_value':
            results['extreme_value'] = self.extreme_value_var(returns, confidence_level)
        
        return {
            'var_estimates': results,
            'model_average': self.calculate_model_average(results),
            'confidence_level': confidence_level,
            'metadata': {
                'data_points': len(returns),
                'period_start': returns.index[0].isoformat(),
                'period_end': returns.index[-1].isoformat()
            }
        }
    
    def historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Historical VaR."""
        return float(returns.quantile(1 - confidence_level))
    
    def parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Parametric VaR (assumes normal distribution)."""
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        return float(mean + z_score * std)
    
    async def monte_carlo_var(
        self,
        returns: pd.Series,
        confidence_level: float,
        simulations: int = 10000
    ) -> float:
        """Calculate Monte Carlo VaR."""
        # Fit distribution parameters
        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Use t-distribution if significant kurtosis
        if abs(kurt) > 1:
            df = 6 / kurt + 4  # Rough approximation
            simulated_returns = stats.t.rvs(df, loc=mean, scale=std, size=simulations)
        else:
            simulated_returns = np.random.normal(mean, std, simulations)
        
        return float(np.percentile(simulated_returns, (1 - confidence_level) * 100))
    
    def extreme_value_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate VaR using Extreme Value Theory."""
        # Use Generalized Pareto Distribution for tail modeling
        threshold = returns.quantile(0.1)  # Use bottom 10% as extreme values
        excesses = returns[returns <= threshold] - threshold
        
        if len(excesses) < 10:  # Need minimum extreme values
            return self.historical_var(returns, confidence_level)
        
        # Fit GPD parameters
        shape, loc, scale = stats.genpareto.fit(excesses)
        
        # Calculate VaR
        n = len(returns)
        n_excesses = len(excesses)
        prob_exceed = n_excesses / n
        
        var_prob = (1 - confidence_level) / prob_exceed
        var_value = threshold + (scale / shape) * (var_prob ** (-shape) - 1)
        
        return float(var_value)
    
    def calculate_model_average(self, results: Dict[str, float]) -> float:
        """Calculate weighted average of VaR estimates."""
        if not results:
            return 0.0
        
        # Simple equal weighting for now
        # In production, you might use more sophisticated weighting
        weights = {method: 1.0 / len(results) for method in results.keys()}
        
        weighted_sum = sum(results[method] * weights[method] for method in results)
        return float(weighted_sum)
```

#### 2. Dynamic Risk Limit Engine

```python
# backend/risk/dynamic_limit_engine.py
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.risk_limit import RiskLimit
from .advanced_var_calculator import AdvancedVaRCalculator


class DynamicLimitEngine:
    """Dynamic risk limit engine with auto-adjustment."""
    
    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis
        self.var_calculator = AdvancedVaRCalculator()
        self.adjustment_strategies = {
            'volatility_based': self.volatility_based_adjustment,
            'var_based': self.var_based_adjustment,
            'regime_based': self.regime_based_adjustment,
            'correlation_based': self.correlation_based_adjustment
        }
    
    async def adjust_limits(
        self,
        portfolio_id: str,
        strategy: str = 'volatility_based',
        **kwargs
    ) -> Dict[str, Any]:
        """Dynamically adjust risk limits."""
        
        # Get current limits
        current_limits = await self.get_portfolio_limits(portfolio_id)
        
        # Get portfolio data for analysis
        portfolio_data = await self.get_portfolio_data(portfolio_id)
        
        if len(portfolio_data) < 30:  # Need minimum data
            return {"message": "Insufficient data for adjustment"}
        
        # Apply adjustment strategy
        adjustment_func = self.adjustment_strategies.get(strategy)
        if not adjustment_func:
            raise ValueError(f"Unknown adjustment strategy: {strategy}")
        
        adjustments = await adjustment_func(portfolio_data, current_limits, **kwargs)
        
        # Apply adjustments
        updated_limits = []
        for limit_id, adjustment in adjustments.items():
            updated_limit = await self.update_limit(limit_id, adjustment)
            updated_limits.append(updated_limit)
        
        # Log adjustment event
        await self.log_adjustment_event(portfolio_id, strategy, adjustments)
        
        return {
            "portfolio_id": portfolio_id,
            "strategy": strategy,
            "adjustments_applied": len(adjustments),
            "updated_limits": updated_limits,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def volatility_based_adjustment(
        self,
        data: pd.DataFrame,
        limits: List[RiskLimit],
        lookback_days: int = 30,
        adjustment_factor: float = 0.1
    ) -> Dict[str, Dict[str, Any]]:
        """Adjust limits based on portfolio volatility changes."""
        
        adjustments = {}
        
        # Calculate current vs historical volatility
        recent_returns = data['returns'].tail(lookback_days)
        historical_returns = data['returns'].tail(lookback_days * 3)  # 3x period for comparison
        
        current_vol = recent_returns.std()
        historical_vol = historical_returns.std()
        
        vol_ratio = current_vol / historical_vol
        
        for limit in limits:
            if limit.limit_type in ['var', 'volatility', 'position_size']:
                # Adjust threshold based on volatility ratio
                if vol_ratio > 1.2:  # Volatility increased significantly
                    new_threshold = limit.threshold_value * (1 - adjustment_factor)
                elif vol_ratio < 0.8:  # Volatility decreased significantly
                    new_threshold = limit.threshold_value * (1 + adjustment_factor)
                else:
                    continue  # No adjustment needed
                
                adjustments[limit.id] = {
                    'new_threshold': new_threshold,
                    'old_threshold': limit.threshold_value,
                    'reason': f'Volatility ratio: {vol_ratio:.3f}',
                    'adjustment_type': 'volatility_based'
                }
        
        return adjustments
    
    async def var_based_adjustment(
        self,
        data: pd.DataFrame,
        limits: List[RiskLimit],
        confidence_level: float = 0.95
    ) -> Dict[str, Dict[str, Any]]:
        """Adjust limits based on VaR calculations."""
        
        adjustments = {}
        
        # Calculate current VaR
        var_result = await self.var_calculator.calculate_var(
            data['returns'],
            method='model_average',
            confidence_level=confidence_level
        )
        
        current_var = var_result['model_average']
        
        for limit in limits:
            if limit.limit_type == 'var':
                # Compare current VaR to limit threshold
                var_utilization = abs(current_var) / limit.threshold_value
                
                if var_utilization > 0.9:  # Close to limit
                    new_threshold = limit.threshold_value * 1.1  # Increase by 10%
                    adjustments[limit.id] = {
                        'new_threshold': new_threshold,
                        'old_threshold': limit.threshold_value,
                        'reason': f'VaR utilization: {var_utilization:.3f}',
                        'adjustment_type': 'var_based'
                    }
        
        return adjustments
    
    async def monitor_and_adjust(self) -> None:
        """Continuous monitoring and adjustment of limits."""
        while True:
            try:
                # Get all portfolios with dynamic limits enabled
                portfolios = await self.get_portfolios_with_dynamic_limits()
                
                for portfolio in portfolios:
                    try:
                        await self.adjust_limits(
                            portfolio['id'],
                            strategy=portfolio.get('adjustment_strategy', 'volatility_based')
                        )
                    except Exception as e:
                        logger.error(f"Error adjusting limits for {portfolio['id']}: {e}")
                
                # Wait before next adjustment cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in limit adjustment cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
```

---

## WebSocket Service Extension

### Custom WebSocket Message Handlers

```python
# backend/websocket/custom_handlers.py
from typing import Dict, Any, List, Optional, Callable
import json
import asyncio
from fastapi import WebSocket
from redis.asyncio import Redis

from .base_handler import BaseWebSocketHandler


class CustomAnalyticsHandler(BaseWebSocketHandler):
    """Custom handler for analytics WebSocket messages."""
    
    def __init__(self, redis: Redis):
        super().__init__("custom_analytics", redis)
        self.subscriptions: Dict[str, set] = {}
        self.message_handlers = {
            'subscribe_analytics': self.handle_subscribe_analytics,
            'unsubscribe_analytics': self.handle_unsubscribe_analytics,
            'request_historical': self.handle_historical_request,
            'configure_alerts': self.handle_configure_alerts
        }
    
    async def handle_message(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ) -> None:
        """Handle incoming WebSocket messages."""
        message_type = message.get('type')
        handler = self.message_handlers.get(message_type)
        
        if handler:
            await handler(websocket, message)
        else:
            await self.send_error(websocket, f"Unknown message type: {message_type}")
    
    async def handle_subscribe_analytics(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ) -> None:
        """Handle analytics subscription requests."""
        portfolio_id = message.get('portfolio_id')
        metrics = message.get('metrics', [])
        
        if not portfolio_id:
            await self.send_error(websocket, "portfolio_id required")
            return
        
        # Add to subscriptions
        connection_id = id(websocket)
        if portfolio_id not in self.subscriptions:
            self.subscriptions[portfolio_id] = set()
        
        self.subscriptions[portfolio_id].add(connection_id)
        
        # Start Redis subscription for this portfolio
        await self.start_portfolio_subscription(portfolio_id)
        
        # Send confirmation
        await websocket.send_text(json.dumps({
            'type': 'subscription_confirmed',
            'portfolio_id': portfolio_id,
            'metrics': metrics
        }))
    
    async def start_portfolio_subscription(self, portfolio_id: str) -> None:
        """Start Redis subscription for portfolio updates."""
        channel = f"analytics:{portfolio_id}"
        
        async def message_processor():
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(channel)
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await self.broadcast_to_subscribers(portfolio_id, data)
                    except Exception as e:
                        logger.error(f"Error processing analytics message: {e}")
        
        # Start processor task
        asyncio.create_task(message_processor())
    
    async def broadcast_to_subscribers(
        self,
        portfolio_id: str,
        data: Dict[str, Any]
    ) -> None:
        """Broadcast analytics data to all subscribers."""
        if portfolio_id not in self.subscriptions:
            return
        
        message = json.dumps({
            'type': 'analytics_update',
            'portfolio_id': portfolio_id,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Send to all subscribers
        disconnected = set()
        for connection_id in self.subscriptions[portfolio_id]:
            websocket = self.get_websocket_by_id(connection_id)
            if websocket:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.warning(f"Failed to send to connection {connection_id}: {e}")
                    disconnected.add(connection_id)
        
        # Clean up disconnected clients
        self.subscriptions[portfolio_id] -= disconnected


class CustomRiskHandler(BaseWebSocketHandler):
    """Custom handler for risk-related WebSocket messages."""
    
    def __init__(self, redis: Redis):
        super().__init__("custom_risk", redis)
        self.alert_subscriptions: Dict[str, Dict[str, Any]] = {}
    
    async def handle_message(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ) -> None:
        """Handle risk-related messages."""
        message_type = message.get('type')
        
        if message_type == 'subscribe_risk_alerts':
            await self.handle_risk_alert_subscription(websocket, message)
        elif message_type == 'configure_risk_thresholds':
            await self.handle_threshold_configuration(websocket, message)
        elif message_type == 'request_risk_snapshot':
            await self.handle_risk_snapshot_request(websocket, message)
    
    async def handle_risk_alert_subscription(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ) -> None:
        """Handle risk alert subscription."""
        portfolio_id = message.get('portfolio_id')
        alert_types = message.get('alert_types', ['all'])
        severity_level = message.get('severity_level', 'warning')
        
        connection_id = id(websocket)
        
        self.alert_subscriptions[connection_id] = {
            'websocket': websocket,
            'portfolio_id': portfolio_id,
            'alert_types': alert_types,
            'severity_level': severity_level,
            'created_at': datetime.utcnow()
        }
        
        # Start monitoring for this subscription
        await self.start_risk_monitoring(connection_id)
    
    async def start_risk_monitoring(self, connection_id: str) -> None:
        """Start risk monitoring for a subscription."""
        subscription = self.alert_subscriptions.get(connection_id)
        if not subscription:
            return
        
        async def risk_monitor():
            while connection_id in self.alert_subscriptions:
                try:
                    # Get current risk metrics
                    portfolio_id = subscription['portfolio_id']
                    risk_data = await self.get_current_risk_metrics(portfolio_id)
                    
                    # Check for alert conditions
                    alerts = self.check_alert_conditions(risk_data, subscription)
                    
                    if alerts:
                        await self.send_risk_alerts(subscription['websocket'], alerts)
                    
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in risk monitoring for {connection_id}: {e}")
                    break
        
        asyncio.create_task(risk_monitor())
    
    def check_alert_conditions(
        self,
        risk_data: Dict[str, Any],
        subscription: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check if any alert conditions are met."""
        alerts = []
        
        # Check VaR threshold
        if 'var_95' in risk_data:
            var_value = risk_data['var_95']
            if abs(var_value) > 50000:  # Example threshold
                alerts.append({
                    'type': 'var_threshold',
                    'severity': 'warning',
                    'message': f'VaR 95% exceeded threshold: {var_value}',
                    'value': var_value
                })
        
        # Check concentration risk
        if 'concentration_risk' in risk_data:
            concentration = risk_data['concentration_risk']
            if concentration > 0.4:  # 40% concentration threshold
                alerts.append({
                    'type': 'concentration_risk',
                    'severity': 'critical',
                    'message': f'High concentration detected: {concentration:.2%}',
                    'value': concentration
                })
        
        return alerts
```

This comprehensive developer guide provides detailed examples and patterns for extending Sprint 3 functionality across all layers of the architecture. The examples show real-world implementations with proper error handling, testing, and integration patterns.