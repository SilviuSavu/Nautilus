"""
Financial ML Models for Core ML Neural Engine Integration
========================================================

Trading-specific neural network architectures optimized for M4 Max Neural Engine
with real-time price prediction, sentiment analysis, and risk assessment capabilities.

Key Features:
- Real-time price prediction models (LSTM, Transformer, CNN)  
- Market sentiment analysis with NLP processing
- Anomaly detection for unusual market patterns
- Technical analysis pattern recognition
- Risk assessment and portfolio optimization models
- High-frequency trading signal generation

Performance Targets:
- < 5ms inference for price predictions
- < 10ms for sentiment analysis  
- 1000+ predictions/second throughput
- Real-time market anomaly detection
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

# Core ML and Neural Engine integration
from .neural_engine_config import neural_performance_context, get_optimization_config
from .coreml_pipeline import convert_model_to_coreml, ModelType, OptimizationLevel

# ML Frameworks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

# Scientific computing
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Text processing for sentiment analysis
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import re
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelArchitecture(Enum):
    """Supported model architectures"""
    LSTM_PRICE_PREDICTOR = "lstm_price_predictor"
    TRANSFORMER_PRICE_PREDICTOR = "transformer_price_predictor"
    CNN_PATTERN_RECOGNIZER = "cnn_pattern_recognizer"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    ANOMALY_DETECTOR = "anomaly_detector"
    RISK_ASSESSOR = "risk_assessor"
    TECHNICAL_INDICATOR_PREDICTOR = "technical_indicator_predictor"
    VOLATILITY_FORECASTER = "volatility_forecaster"

class DataFrequency(Enum):
    """Data frequency for models"""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAILY = "daily"

@dataclass
class ModelConfig:
    """Configuration for financial ML models"""
    architecture: ModelArchitecture
    sequence_length: int
    input_features: int
    output_features: int
    hidden_size: int
    num_layers: int
    dropout_rate: float
    learning_rate: float
    batch_size: int
    epochs: int
    data_frequency: DataFrequency
    lookback_period: int
    prediction_horizon: int

@dataclass
class TrainingResult:
    """Result of model training"""
    success: bool
    model_path: Optional[str] = None
    coreml_model_path: Optional[str] = None
    training_time_ms: float = 0.0
    validation_metrics: Optional[Dict[str, float]] = None
    training_loss: Optional[List[float]] = None
    validation_loss: Optional[List[float]] = None
    error_message: Optional[str] = None

@dataclass
class PredictionResult:
    """Result of model prediction"""
    success: bool
    predictions: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    inference_time_ms: float = 0.0
    model_version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class LSTMPricePredictor(nn.Module):
    """LSTM-based price prediction model optimized for Neural Engine"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2,
                 sequence_length: int = 60):
        super(LSTMPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # LSTM layers optimized for Neural Engine
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism for improved accuracy
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last time step output
        last_output = attn_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        
        return out

class TransformerPricePredictor(nn.Module):
    """Transformer-based price prediction model with positional encoding"""
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 output_size: int = 1,
                 dropout: float = 0.1,
                 sequence_length: int = 100):
        super(TransformerPricePredictor, self).__init__()
        
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer
        transformer_out = self.transformer(x)
        
        # Use last time step for prediction
        last_output = transformer_out[:, -1, :]
        
        # Apply layer norm and dropout
        out = self.layer_norm(last_output)
        out = self.dropout(out)
        
        # Final prediction
        out = self.output_projection(out)
        
        return out

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class CNNPatternRecognizer(nn.Module):
    """CNN-based technical pattern recognition model"""
    
    def __init__(self,
                 input_channels: int = 5,  # OHLCV
                 sequence_length: int = 60,
                 num_patterns: int = 10,  # Number of patterns to recognize
                 dropout: float = 0.2):
        super(CNNPatternRecognizer, self).__init__()
        
        self.sequence_length = sequence_length
        
        # 1D Convolutional layers for time series pattern recognition
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(16)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_patterns)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))  # Multi-label classification
        
        return x

class SentimentAnalyzer(nn.Module):
    """LSTM-based sentiment analyzer for financial news and social media"""
    
    def __init__(self,
                 vocab_size: int = 10000,
                 embedding_dim: int = 100,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 3,  # Negative, Neutral, Positive
                 dropout: float = 0.3):
        super(SentimentAnalyzer, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(embedded)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global max pooling
        pooled = torch.max(attn_out, dim=1)[0]
        
        # Classification
        output = self.classifier(pooled)
        probabilities = self.softmax(output)
        
        return probabilities

class AnomalyDetector:
    """Isolation Forest-based market anomaly detector"""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Scikit-learn not available")
            
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'AnomalyDetector':
        """Fit the anomaly detector"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies
        
        Returns:
            Tuple of (anomaly_labels, anomaly_scores)
            anomaly_labels: -1 for anomalies, 1 for normal
            anomaly_scores: Lower scores indicate higher anomaly likelihood
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        
        return labels, scores

class VolatilityForecaster(nn.Module):
    """LSTM-based volatility forecasting model"""
    
    def __init__(self,
                 input_size: int = 5,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.2):
        super(VolatilityForecaster, self).__init__()
        
        # LSTM for volatility pattern learning
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers with proper scaling for volatility
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softplus()  # Ensure positive volatility predictions
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last time step
        last_output = lstm_out[:, -1, :]
        volatility = self.fc(last_output)
        return volatility

class FinancialModelBuilder:
    """Builder class for financial ML models with Neural Engine optimization"""
    
    def __init__(self):
        self.models_cache = {}
        self.training_history = {}
        self.data_preprocessors = {}
        
    async def build_lstm_price_predictor(self, config: ModelConfig) -> Tuple[nn.Module, str]:
        """
        Build LSTM price prediction model
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (model, model_identifier)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        logger.info("Building LSTM price predictor")
        
        model = LSTMPricePredictor(
            input_size=config.input_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_features,
            dropout=config.dropout_rate,
            sequence_length=config.sequence_length
        )
        
        # Generate model identifier
        model_id = f"lstm_price_{config.hidden_size}h_{config.num_layers}l_{int(time.time())}"
        
        # Cache the model
        self.models_cache[model_id] = {
            'model': model,
            'config': config,
            'created_at': time.time()
        }
        
        return model, model_id
    
    async def build_transformer_price_predictor(self, config: ModelConfig) -> Tuple[nn.Module, str]:
        """Build Transformer price prediction model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        logger.info("Building Transformer price predictor")
        
        model = TransformerPricePredictor(
            input_size=config.input_features,
            d_model=config.hidden_size,
            nhead=8,
            num_layers=config.num_layers,
            output_size=config.output_features,
            dropout=config.dropout_rate,
            sequence_length=config.sequence_length
        )
        
        model_id = f"transformer_price_{config.hidden_size}d_{config.num_layers}l_{int(time.time())}"
        
        self.models_cache[model_id] = {
            'model': model,
            'config': config,
            'created_at': time.time()
        }
        
        return model, model_id
    
    async def build_cnn_pattern_recognizer(self, config: ModelConfig) -> Tuple[nn.Module, str]:
        """Build CNN pattern recognition model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        logger.info("Building CNN pattern recognizer")
        
        model = CNNPatternRecognizer(
            input_channels=config.input_features,
            sequence_length=config.sequence_length,
            num_patterns=config.output_features,
            dropout=config.dropout_rate
        )
        
        model_id = f"cnn_pattern_{config.input_features}ch_{config.sequence_length}seq_{int(time.time())}"
        
        self.models_cache[model_id] = {
            'model': model,
            'config': config,
            'created_at': time.time()
        }
        
        return model, model_id
    
    async def build_sentiment_analyzer(self, config: ModelConfig, vocab_size: int = 10000) -> Tuple[nn.Module, str]:
        """Build sentiment analysis model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        logger.info("Building sentiment analyzer")
        
        model = SentimentAnalyzer(
            vocab_size=vocab_size,
            embedding_dim=config.hidden_size // 2,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_features,
            dropout=config.dropout_rate
        )
        
        model_id = f"sentiment_{vocab_size}vocab_{config.hidden_size}h_{int(time.time())}"
        
        self.models_cache[model_id] = {
            'model': model,
            'config': config,
            'created_at': time.time()
        }
        
        return model, model_id
    
    async def build_anomaly_detector(self, config: ModelConfig) -> Tuple[AnomalyDetector, str]:
        """Build anomaly detection model"""
        logger.info("Building anomaly detector")
        
        model = AnomalyDetector(
            contamination=0.1,  # Expect 10% anomalies
            n_estimators=100
        )
        
        model_id = f"anomaly_detector_{int(time.time())}"
        
        self.models_cache[model_id] = {
            'model': model,
            'config': config,
            'created_at': time.time()
        }
        
        return model, model_id
    
    async def build_volatility_forecaster(self, config: ModelConfig) -> Tuple[nn.Module, str]:
        """Build volatility forecasting model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
            
        logger.info("Building volatility forecaster")
        
        model = VolatilityForecaster(
            input_size=config.input_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_features,
            dropout=config.dropout_rate
        )
        
        model_id = f"volatility_{config.hidden_size}h_{config.num_layers}l_{int(time.time())}"
        
        self.models_cache[model_id] = {
            'model': model,
            'config': config,
            'created_at': time.time()
        }
        
        return model, model_id

class FinancialModelTrainer:
    """Trainer for financial ML models with Neural Engine optimization"""
    
    def __init__(self):
        self.training_metrics = {}
        self.best_models = {}
        
    async def train_pytorch_model(self,
                                model: nn.Module,
                                train_data: DataLoader,
                                val_data: Optional[DataLoader] = None,
                                config: ModelConfig = None,
                                model_id: str = None) -> TrainingResult:
        """
        Train PyTorch model with optimization for Neural Engine deployment
        
        Args:
            model: PyTorch model to train
            train_data: Training data loader
            val_data: Validation data loader
            config: Model configuration
            model_id: Model identifier
            
        Returns:
            TrainingResult with training details
        """
        if not TORCH_AVAILABLE:
            return TrainingResult(
                success=False,
                error_message="PyTorch not available"
            )
        
        start_time = time.perf_counter()
        training_losses = []
        validation_losses = []
        
        try:
            logger.info(f"Training PyTorch model: {model_id}")
            
            # Setup training
            device = torch.device("cpu")  # Use CPU for better Core ML compatibility
            model = model.to(device)
            
            # Configure optimizer and loss function
            if config and config.architecture in [ModelArchitecture.LSTM_PRICE_PREDICTOR, 
                                                 ModelArchitecture.TRANSFORMER_PRICE_PREDICTOR,
                                                 ModelArchitecture.VOLATILITY_FORECASTER]:
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
            elif config and config.architecture == ModelArchitecture.CNN_PATTERN_RECOGNIZER:
                criterion = nn.BCELoss()  # Multi-label classification
                optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
            elif config and config.architecture == ModelArchitecture.SENTIMENT_ANALYZER:
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
            else:
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            epochs = config.epochs if config else 50
            best_val_loss = float('inf')
            early_stopping_patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                epoch_train_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_data):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_train_loss += loss.item()
                
                avg_train_loss = epoch_train_loss / len(train_data)
                training_losses.append(avg_train_loss)
                
                # Validation phase
                if val_data:
                    model.eval()
                    epoch_val_loss = 0.0
                    
                    with torch.no_grad():
                        for data, target in val_data:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            loss = criterion(output, target)
                            epoch_val_loss += loss.item()
                    
                    avg_val_loss = epoch_val_loss / len(val_data)
                    validation_losses.append(avg_val_loss)
                    
                    # Learning rate scheduling
                    scheduler.step(avg_val_loss)
                    
                    # Early stopping
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        # Save best model state
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                    
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                else:
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")
            
            # Restore best model state if validation was used
            if val_data and 'best_model_state' in locals():
                model.load_state_dict(best_model_state)
            
            # Save trained model
            model_path = f"/tmp/nautilus_models/{model_id}_trained.pth"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
            # Convert to Core ML for Neural Engine deployment
            coreml_model_path = f"/tmp/nautilus_models/{model_id}_coreml.mlpackage"
            
            # Create example input for conversion
            sample_batch = next(iter(train_data))
            example_input = sample_batch[0][:1]  # Take first sample
            
            conversion_result = await convert_model_to_coreml(
                model=model,
                example_input=example_input,
                model_type=ModelType.PYTORCH,
                output_path=coreml_model_path,
                optimization_level=OptimizationLevel.BALANCED
            )
            
            # Calculate validation metrics
            validation_metrics = {}
            if val_data:
                validation_metrics = await self._calculate_validation_metrics(
                    model, val_data, criterion, device, config
                )
            
            training_time_ms = (time.perf_counter() - start_time) * 1000
            
            result = TrainingResult(
                success=True,
                model_path=model_path,
                coreml_model_path=coreml_model_path if conversion_result.success else None,
                training_time_ms=training_time_ms,
                validation_metrics=validation_metrics,
                training_loss=training_losses,
                validation_loss=validation_losses if val_data else None
            )
            
            logger.info(f"Model training completed in {training_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            training_time_ms = (time.perf_counter() - start_time) * 1000
            
            return TrainingResult(
                success=False,
                training_time_ms=training_time_ms,
                error_message=str(e)
            )
    
    async def _calculate_validation_metrics(self,
                                          model: nn.Module,
                                          val_data: DataLoader,
                                          criterion: nn.Module,
                                          device: torch.device,
                                          config: Optional[ModelConfig] = None) -> Dict[str, float]:
        """Calculate comprehensive validation metrics"""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                all_predictions.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_data)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        metrics = {'validation_loss': avg_loss}
        
        # Calculate metrics based on model architecture
        if config and config.architecture in [
            ModelArchitecture.LSTM_PRICE_PREDICTOR,
            ModelArchitecture.TRANSFORMER_PRICE_PREDICTOR,
            ModelArchitecture.VOLATILITY_FORECASTER
        ]:
            # Regression metrics
            mse = np.mean((all_predictions - all_targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(all_predictions - all_targets))
            
            # R-squared
            ss_res = np.sum((all_targets - all_predictions) ** 2)
            ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            metrics.update({
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            })
            
        elif config and config.architecture in [
            ModelArchitecture.CNN_PATTERN_RECOGNIZER,
            ModelArchitecture.SENTIMENT_ANALYZER
        ]:
            # Classification metrics
            if config.architecture == ModelArchitecture.CNN_PATTERN_RECOGNIZER:
                # Multi-label classification
                predictions_binary = (all_predictions > 0.5).astype(int)
                targets_binary = (all_targets > 0.5).astype(int)
                
                accuracy = np.mean(predictions_binary == targets_binary)
                metrics['accuracy'] = accuracy
                
            else:
                # Multi-class classification
                predicted_classes = np.argmax(all_predictions, axis=1)
                target_classes = np.argmax(all_targets, axis=1) if all_targets.ndim > 1 else all_targets
                
                if SKLEARN_AVAILABLE:
                    accuracy = accuracy_score(target_classes, predicted_classes)
                    precision = precision_score(target_classes, predicted_classes, average='weighted', zero_division=0)
                    recall = recall_score(target_classes, predicted_classes, average='weighted', zero_division=0)
                    f1 = f1_score(target_classes, predicted_classes, average='weighted', zero_division=0)
                    
                    metrics.update({
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    })
        
        return metrics

class DataPreprocessor:
    """Data preprocessing pipeline for financial ML models"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = {}
        self.preprocessing_stats = {}
    
    def prepare_price_data(self, 
                          df: pd.DataFrame,
                          sequence_length: int = 60,
                          prediction_horizon: int = 1,
                          features: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare price data for time series prediction
        
        Args:
            df: DataFrame with OHLCV data
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
            features: List of feature columns to use
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if features is None:
            features = ['open', 'high', 'low', 'close', 'volume']
        
        # Ensure required columns exist
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            raise ValueError("No valid features found in DataFrame")
        
        # Calculate technical indicators
        df = self._add_technical_indicators(df)
        
        # Update features to include technical indicators
        tech_indicators = ['sma_20', 'ema_12', 'ema_26', 'rsi', 'macd']
        available_features.extend([f for f in tech_indicators if f in df.columns])
        
        # Normalize features
        scaler = MinMaxScaler() if SKLEARN_AVAILABLE else None
        if scaler:
            scaled_data = scaler.fit_transform(df[available_features])
            self.scalers['price_scaler'] = scaler
        else:
            scaled_data = df[available_features].values
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data) - prediction_horizon + 1):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i+prediction_horizon-1, available_features.index('close')])
        
        X = np.array(X)
        y = np.array(y)
        
        self.feature_columns['price_features'] = available_features
        
        logger.info(f"Prepared price data: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def prepare_sentiment_data(self,
                             texts: List[str],
                             labels: List[int],
                             max_sequence_length: int = 100,
                             vocab_size: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare text data for sentiment analysis
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels
            max_sequence_length: Maximum sequence length
            vocab_size: Vocabulary size
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, using basic preprocessing")
            
        # Preprocess texts
        processed_texts = []
        for text in texts:
            processed = self._preprocess_text(text)
            processed_texts.append(processed)
        
        # Create vocabulary
        word_counts = {}
        for text in processed_texts:
            for word in text:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Build vocabulary with most common words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab = {word: i+1 for i, (word, _) in enumerate(sorted_words[:vocab_size-1])}
        vocab['<UNK>'] = vocab_size
        
        # Convert texts to sequences
        X = []
        for text in processed_texts:
            sequence = [vocab.get(word, vocab_size) for word in text]
            # Pad or truncate to fixed length
            if len(sequence) < max_sequence_length:
                sequence.extend([0] * (max_sequence_length - len(sequence)))
            else:
                sequence = sequence[:max_sequence_length]
            X.append(sequence)
        
        X = np.array(X)
        y = np.array(labels)
        
        # Store vocabulary for later use
        self.feature_columns['sentiment_vocab'] = vocab
        
        logger.info(f"Prepared sentiment data: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def prepare_anomaly_data(self, df: pd.DataFrame, features: List[str] = None) -> np.ndarray:
        """
        Prepare data for anomaly detection
        
        Args:
            df: DataFrame with market data
            features: List of features to use
            
        Returns:
            Prepared feature matrix
        """
        if features is None:
            features = ['volume', 'price_change', 'volatility']
        
        # Calculate additional features for anomaly detection
        df = df.copy()
        if 'close' in df.columns:
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['close'].rolling(window=20).std()
        
        if 'volume' in df.columns and 'volume' not in features:
            features.append('volume')
        
        # Select available features
        available_features = [f for f in features if f in df.columns]
        
        # Remove NaN values
        df = df[available_features].dropna()
        
        # Scale features
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            X = scaler.fit_transform(df[available_features])
            self.scalers['anomaly_scaler'] = scaler
        else:
            X = df[available_features].values
        
        logger.info(f"Prepared anomaly data: X shape {X.shape}")
        return X
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        df = df.copy()
        
        if 'close' in df.columns:
            # Simple Moving Average
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        return df.dropna()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for sentiment analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        if NLTK_AVAILABLE:
            try:
                # Tokenize
                tokens = word_tokenize(text)
                
                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
                
                # Lemmatization
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
                
                return tokens
            except:
                # Fallback to simple tokenization
                return text.split()
        else:
            # Simple tokenization
            return text.split()

# Global instances
model_builder = FinancialModelBuilder()
model_trainer = FinancialModelTrainer()
data_preprocessor = DataPreprocessor()

# Convenience functions
async def create_price_prediction_model(config: ModelConfig) -> Tuple[nn.Module, str]:
    """Create optimized price prediction model"""
    if config.architecture == ModelArchitecture.LSTM_PRICE_PREDICTOR:
        return await model_builder.build_lstm_price_predictor(config)
    elif config.architecture == ModelArchitecture.TRANSFORMER_PRICE_PREDICTOR:
        return await model_builder.build_transformer_price_predictor(config)
    else:
        raise ValueError(f"Unsupported architecture: {config.architecture}")

async def create_pattern_recognition_model(config: ModelConfig) -> Tuple[nn.Module, str]:
    """Create optimized pattern recognition model"""
    return await model_builder.build_cnn_pattern_recognizer(config)

async def create_sentiment_model(config: ModelConfig) -> Tuple[nn.Module, str]:
    """Create optimized sentiment analysis model"""
    return await model_builder.build_sentiment_analyzer(config)

def get_default_model_config(architecture: ModelArchitecture, 
                           data_frequency: DataFrequency = DataFrequency.MINUTE) -> ModelConfig:
    """Get default configuration for model architecture"""
    configs = {
        ModelArchitecture.LSTM_PRICE_PREDICTOR: ModelConfig(
            architecture=architecture,
            sequence_length=60,
            input_features=8,  # OHLCV + technical indicators
            output_features=1,
            hidden_size=128,
            num_layers=2,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            data_frequency=data_frequency,
            lookback_period=60,
            prediction_horizon=1
        ),
        ModelArchitecture.TRANSFORMER_PRICE_PREDICTOR: ModelConfig(
            architecture=architecture,
            sequence_length=100,
            input_features=8,
            output_features=1,
            hidden_size=256,
            num_layers=6,
            dropout_rate=0.1,
            learning_rate=0.0001,
            batch_size=16,
            epochs=50,
            data_frequency=data_frequency,
            lookback_period=100,
            prediction_horizon=1
        ),
        ModelArchitecture.CNN_PATTERN_RECOGNIZER: ModelConfig(
            architecture=architecture,
            sequence_length=60,
            input_features=5,  # OHLCV
            output_features=10,  # Number of patterns
            hidden_size=256,
            num_layers=3,
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=32,
            epochs=80,
            data_frequency=data_frequency,
            lookback_period=60,
            prediction_horizon=1
        ),
        ModelArchitecture.SENTIMENT_ANALYZER: ModelConfig(
            architecture=architecture,
            sequence_length=100,
            input_features=10000,  # Vocab size
            output_features=3,  # Negative, Neutral, Positive
            hidden_size=128,
            num_layers=2,
            dropout_rate=0.3,
            learning_rate=0.001,
            batch_size=64,
            epochs=20,
            data_frequency=DataFrequency.DAILY,
            lookback_period=1,
            prediction_horizon=1
        )
    }
    
    return configs.get(architecture, configs[ModelArchitecture.LSTM_PRICE_PREDICTOR])

def get_trading_models_status() -> Dict[str, Any]:
    """Get comprehensive trading models status"""
    return {
        'frameworks_available': {
            'pytorch': TORCH_AVAILABLE,
            'tensorflow': TENSORFLOW_AVAILABLE,
            'sklearn': SKLEARN_AVAILABLE,
            'nltk': NLTK_AVAILABLE
        },
        'cached_models': len(model_builder.models_cache),
        'supported_architectures': [arch.value for arch in ModelArchitecture],
        'data_frequencies': [freq.value for freq in DataFrequency],
        'preprocessing_stats': data_preprocessor.preprocessing_stats,
        'training_metrics': model_trainer.training_metrics
    }