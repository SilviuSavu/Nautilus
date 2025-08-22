# 🛠️ Implementation Guides - Nautilus Trading Platform

## Guide 1: Live Trading Engine Implementation

### Architecture Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Order Entry   │ → │ Order Management │ → │   Execution     │
│     System      │    │     System       │    │    Engine       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Risk Engine    │    │ Position Keeper │    │ Market Data     │
│                 │    │                 │    │   Gateway       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components Implementation

#### 1. Order Management System (OMS)
```python
# File: backend/trading_engine/order_management.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime
import asyncio
import uuid

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    client_order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = self.created_at

class OrderManagementSystem:
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.order_callbacks: List[callable] = []
        
    async def submit_order(self, order: Order) -> str:
        """Submit order for execution"""
        if not order.id:
            order.id = str(uuid.uuid4())
        
        # Risk checks
        if not await self._pre_trade_risk_check(order):
            order.status = OrderStatus.REJECTED
            await self._notify_order_update(order)
            raise ValueError("Order rejected by risk engine")
        
        # Store order
        self.orders[order.id] = order
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.now()
        
        # Send to execution engine
        await self._route_to_execution(order)
        await self._notify_order_update(order)
        
        return order.id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        await self._notify_order_update(order)
        
        return True
    
    async def _pre_trade_risk_check(self, order: Order) -> bool:
        """Pre-trade risk validation"""
        # Implement risk checks here
        return True
    
    async def _route_to_execution(self, order: Order):
        """Route order to appropriate execution venue"""
        # Implement execution routing logic
        pass
    
    async def _notify_order_update(self, order: Order):
        """Notify all callbacks of order updates"""
        for callback in self.order_callbacks:
            await callback(order)
```

#### 2. Execution Engine
```python
# File: backend/trading_engine/execution_engine.py

import asyncio
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

class ExecutionVenue(ABC):
    @abstractmethod
    async def submit_order(self, order: Order) -> bool:
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict:
        pass

class IBKRVenue(ExecutionVenue):
    def __init__(self, ib_client):
        self.ib_client = ib_client
    
    async def submit_order(self, order: Order) -> bool:
        # Implement IBKR order submission
        try:
            # Convert internal order to IBKR format
            ib_order = self._convert_to_ib_order(order)
            result = await self.ib_client.place_order(ib_order)
            return result.success
        except Exception as e:
            logging.error(f"IBKR order submission failed: {e}")
            return False
    
    def _convert_to_ib_order(self, order: Order):
        # Implementation details for IBKR order conversion
        pass

class SmartOrderRouter:
    def __init__(self):
        self.venues: List[ExecutionVenue] = []
        self.routing_rules: Dict = {}
    
    def add_venue(self, venue: ExecutionVenue):
        self.venues.append(venue)
    
    async def route_order(self, order: Order) -> bool:
        """Route order to best venue based on current market conditions"""
        best_venue = await self._select_best_venue(order)
        return await best_venue.submit_order(order)
    
    async def _select_best_venue(self, order: Order) -> ExecutionVenue:
        """Select optimal execution venue"""
        # Implement venue selection logic based on:
        # - Liquidity
        # - Spreads
        # - Historical fill rates
        # - Order size
        return self.venues[0]  # Simplified for now

class ExecutionEngine:
    def __init__(self):
        self.router = SmartOrderRouter()
        self.active_orders: Dict[str, Order] = {}
        
    async def execute_order(self, order: Order):
        """Execute order through smart order routing"""
        self.active_orders[order.id] = order
        
        success = await self.router.route_order(order)
        if not success:
            order.status = OrderStatus.REJECTED
            await self._handle_execution_failure(order)
```

#### 3. Real-time Risk Engine
```python
# File: backend/trading_engine/risk_engine.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio

@dataclass
class RiskLimits:
    max_position_size: float
    max_order_size: float
    max_daily_loss: float
    max_portfolio_leverage: float
    allowed_symbols: List[str]

@dataclass
class RiskMetrics:
    current_exposure: float
    daily_pnl: float
    portfolio_leverage: float
    var_95: float
    
class RealTimeRiskEngine:
    def __init__(self):
        self.risk_limits: Dict[str, RiskLimits] = {}
        self.current_positions: Dict[str, float] = {}
        self.daily_pnl: float = 0.0
        
    async def check_pre_trade_risk(self, order: Order, portfolio_id: str) -> bool:
        """Pre-trade risk check"""
        limits = self.risk_limits.get(portfolio_id)
        if not limits:
            return False
            
        # Check order size limit
        if order.quantity > limits.max_order_size:
            return False
            
        # Check position limit
        current_pos = self.current_positions.get(order.symbol, 0)
        new_position = current_pos + (order.quantity if order.side == OrderSide.BUY else -order.quantity)
        
        if abs(new_position) > limits.max_position_size:
            return False
            
        # Check symbol allowlist
        if order.symbol not in limits.allowed_symbols:
            return False
            
        return True
    
    async def calculate_portfolio_risk(self, portfolio_id: str) -> RiskMetrics:
        """Calculate real-time portfolio risk metrics"""
        # Implement VaR calculation, exposure analysis, etc.
        return RiskMetrics(
            current_exposure=sum(abs(pos) for pos in self.current_positions.values()),
            daily_pnl=self.daily_pnl,
            portfolio_leverage=1.5,  # Calculate actual leverage
            var_95=100000.0  # Calculate actual VaR
        )
```

### Performance Requirements
- **Order Latency**: <100ms end-to-end
- **Throughput**: 1000+ orders/second
- **Availability**: 99.99% during market hours
- **Data Consistency**: ACID compliance for all transactions

---

## Guide 2: AI/ML Factor Research Platform

### ML Pipeline Architecture
```
Data Sources → Feature Engineering → Model Training → Model Serving → Factor Calculation
     │               │                    │              │               │
     ▼               ▼                    ▼              ▼               ▼
Market Data     Automated           Distributed     Real-time       Factor
Alt Data        Feature             ML Training     Inference       Database
News/Social     Selection           Validation      <1ms            Time Series
```

### Implementation Components

#### 1. Data Pipeline
```python
# File: backend/ml_platform/data_pipeline.py

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

@dataclass
class DataSource:
    name: str
    source_type: str  # market_data, news, social, fundamental
    update_frequency: str  # realtime, daily, hourly
    schema: Dict[str, Any]

class DataConnector(ABC):
    @abstractmethod
    async def fetch_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        pass
    
    @abstractmethod
    async def stream_data(self) -> AsyncIterator[Dict]:
        pass

class MarketDataConnector(DataConnector):
    def __init__(self, ib_client):
        self.ib_client = ib_client
    
    async def fetch_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        # Fetch historical market data
        pass
    
    async def stream_data(self) -> AsyncIterator[Dict]:
        # Stream real-time market data
        async for tick in self.ib_client.stream_ticks():
            yield {
                'symbol': tick.symbol,
                'price': tick.price,
                'volume': tick.volume,
                'timestamp': tick.timestamp
            }

class NewsDataConnector(DataConnector):
    def __init__(self, news_api_key: str):
        self.api_key = news_api_key
    
    async def fetch_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        # Fetch news data from various sources
        pass

class DataPipeline:
    def __init__(self):
        self.connectors: Dict[str, DataConnector] = {}
        self.processors: List[DataProcessor] = []
        
    def add_connector(self, name: str, connector: DataConnector):
        self.connectors[name] = connector
    
    async def process_batch_data(self, source: str, start_time: datetime, end_time: datetime):
        """Process batch data for training"""
        connector = self.connectors[source]
        raw_data = await connector.fetch_data(start_time, end_time)
        
        # Apply data processors
        processed_data = raw_data
        for processor in self.processors:
            processed_data = await processor.process(processed_data)
        
        return processed_data
    
    async def stream_real_time_data(self, source: str):
        """Stream and process real-time data"""
        connector = self.connectors[source]
        async for data_point in connector.stream_data():
            # Process real-time data point
            yield await self._process_real_time_point(data_point)
```

#### 2. Feature Engineering Engine
```python
# File: backend/ml_platform/feature_engine.py

import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Any
from dataclasses import dataclass

@dataclass
class FeatureDefinition:
    name: str
    calculation_func: Callable
    dependencies: List[str]
    lookback_period: int
    update_frequency: str

class FeatureEngine:
    def __init__(self):
        self.features: Dict[str, FeatureDefinition] = {}
        self.feature_cache: Dict[str, pd.Series] = {}
        
    def register_feature(self, feature: FeatureDefinition):
        """Register a new feature calculation"""
        self.features[feature.name] = feature
    
    async def calculate_features(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Calculate all features for given symbols"""
        results = {}
        
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol]
            symbol_features = {}
            
            for feature_name, feature_def in self.features.items():
                feature_value = await self._calculate_single_feature(
                    feature_def, symbol_data, symbol
                )
                symbol_features[feature_name] = feature_value
            
            results[symbol] = symbol_features
        
        return pd.DataFrame(results).T
    
    async def _calculate_single_feature(self, feature_def: FeatureDefinition, 
                                      data: pd.DataFrame, symbol: str) -> float:
        """Calculate a single feature value"""
        try:
            return feature_def.calculation_func(data)
        except Exception as e:
            logging.error(f"Feature calculation failed for {feature_def.name}: {e}")
            return np.nan

# Technical indicators as features
def calculate_rsi(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate RSI indicator"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_bollinger_position(data: pd.DataFrame, period: int = 20) -> float:
    """Calculate position within Bollinger Bands"""
    close = data['close']
    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper_band = ma + (2 * std)
    lower_band = ma - (2 * std)
    
    current_price = close.iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    
    return (current_price - current_lower) / (current_upper - current_lower)

# Fundamental features
def calculate_pe_ratio(data: pd.DataFrame) -> float:
    """Calculate P/E ratio from fundamental data"""
    if 'earnings_per_share' in data.columns and 'price' in data.columns:
        eps = data['earnings_per_share'].iloc[-1]
        price = data['price'].iloc[-1]
        return price / eps if eps != 0 else np.nan
    return np.nan

# Register features
feature_engine = FeatureEngine()

feature_engine.register_feature(FeatureDefinition(
    name="rsi_14",
    calculation_func=lambda x: calculate_rsi(x, 14),
    dependencies=["close"],
    lookback_period=14,
    update_frequency="daily"
))

feature_engine.register_feature(FeatureDefinition(
    name="bollinger_position",
    calculation_func=calculate_bollinger_position,
    dependencies=["close"],
    lookback_period=20,
    update_frequency="daily"
))
```

#### 3. Model Training Framework
```python
# File: backend/ml_platform/model_training.py

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from typing import Dict, Any, List, Tuple

class ModelTrainer:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    async def train_factor_model(self, 
                               features: pd.DataFrame, 
                               targets: pd.DataFrame,
                               model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a factor prediction model"""
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(model_config)
            
            # Prepare data
            X, y = self._prepare_training_data(features, targets)
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train model
            model = self._create_model(model_config)
            
            # Cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                
                mse = mean_squared_error(y_val, val_pred)
                cv_scores.append(mse)
            
            # Train final model on all data
            model.fit(X, y)
            
            # Log metrics
            mlflow.log_metric("cv_mse_mean", np.mean(cv_scores))
            mlflow.log_metric("cv_mse_std", np.std(cv_scores))
            
            # Log model
            model_path = f"models/{self.experiment_name}/model.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            
            return {
                "model": model,
                "cv_scores": cv_scores,
                "feature_importance": self._get_feature_importance(model, X.columns)
            }
    
    def _prepare_training_data(self, features: pd.DataFrame, targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare and align training data"""
        # Align features and targets by timestamp
        aligned_data = pd.merge(features, targets, left_index=True, right_index=True, how='inner')
        
        # Handle missing values
        aligned_data = aligned_data.dropna()
        
        X = aligned_data[features.columns]
        y = aligned_data[targets.columns[0]]  # Assuming single target
        
        return X, y
    
    def _create_model(self, config: Dict[str, Any]):
        """Create model based on configuration"""
        model_type = config.get("model_type", "random_forest")
        
        if model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", 10),
                random_state=42
            )
        # Add other model types as needed
        
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        return {}
```

### Performance Requirements
- **Feature Calculation**: <1ms per feature per symbol
- **Model Training**: Complete within 4 hours for full dataset
- **Model Inference**: <10ms for real-time scoring
- **Data Processing**: 10TB+ daily data processing capability

---

## Guide 3: Production Deployment Infrastructure

### Kubernetes Architecture
```yaml
# File: deployment/kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: nautilus-prod
  labels:
    name: nautilus-prod
    environment: production

---
# File: deployment/kubernetes/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nautilus-backend
  namespace: nautilus-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nautilus-backend
  template:
    metadata:
      labels:
        app: nautilus-backend
    spec:
      containers:
      - name: backend
        image: nautilus/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: nautilus-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: nautilus-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# File: deployment/kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: nautilus-backend-service
  namespace: nautilus-prod
spec:
  selector:
    app: nautilus-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# File: deployment/kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nautilus-ingress
  namespace: nautilus-prod
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.nautilus-trading.com
    secretName: nautilus-tls
  rules:
  - host: api.nautilus-trading.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nautilus-backend-service
            port:
              number: 80
```

### Terraform Infrastructure
```hcl
# File: infrastructure/terraform/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "nautilus-prod"
  cluster_version = "1.24"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  node_groups = {
    main = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 3
      
      instance_types = ["c5.2xlarge"]
      
      k8s_labels = {
        Environment = "production"
        Application = "nautilus"
      }
    }
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "main" {
  identifier = "nautilus-prod-db"
  
  engine            = "postgres"
  engine_version    = "14.9"
  instance_class    = "db.r5.2xlarge"
  allocated_storage = 1000
  storage_type      = "gp3"
  storage_encrypted = true
  
  db_name  = "nautilus"
  username = "nautilus_admin"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "nautilus-prod-final-snapshot"
  
  tags = {
    Name        = "nautilus-prod-db"
    Environment = "production"
  }
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "nautilus-prod-redis"
  description                = "Redis cluster for Nautilus production"
  
  node_type                  = "cache.r6g.2xlarge"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 3
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name        = "nautilus-prod-redis"
    Environment = "production"
  }
}
```

### CI/CD Pipeline
```yaml
# File: .github/workflows/deploy-production.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  release:
    types: [published]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=backend --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push Docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: nautilus-backend
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Update kubeconfig
      run: aws eks update-kubeconfig --region us-east-1 --name nautilus-prod
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/nautilus-backend backend=$ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }} -n nautilus-prod
        kubectl rollout status deployment/nautilus-backend -n nautilus-prod
```

This comprehensive implementation plan provides detailed guidance for all four objectives, with specific code examples, architecture diagrams, and deployment configurations. Each guide includes performance requirements and can be implemented incrementally.