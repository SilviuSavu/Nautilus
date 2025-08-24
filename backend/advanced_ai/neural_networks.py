"""
Advanced Neural Networks for Multi-Modal Market Data Processing
Implementation of attention mechanisms, transformers, and multi-modal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
import math
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class MarketDataSequence:
    """Multi-modal market data sequence"""
    prices: torch.Tensor       # OHLCV data
    technical: torch.Tensor    # Technical indicators
    fundamental: torch.Tensor  # Fundamental data
    sentiment: torch.Tensor    # Sentiment scores
    alternative: torch.Tensor  # Alternative data
    timestamps: List[datetime]
    symbols: List[str]
    metadata: Dict[str, Any]


@dataclass
class NeuralPrediction:
    """Neural network prediction result"""
    predicted_values: torch.Tensor
    confidence_scores: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    feature_importance: Optional[Dict[str, float]] = None
    prediction_intervals: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    explanation: Optional[str] = None


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                           (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiModalFusionBlock(nn.Module):
    """Multi-modal fusion block for combining different data types"""
    
    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 256, 
                 output_dim: int = 128, fusion_method: str = 'attention'):
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        
        # Individual projection layers for each modality
        self.projectors = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.projectors[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Fusion layers
        if fusion_method == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=8, dropout=0.1
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)
        elif fusion_method == 'concatenation':
            total_dim = len(input_dims) * hidden_dim
            self.fusion_linear = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        elif fusion_method == 'weighted_sum':
            self.modality_weights = nn.Parameter(
                torch.ones(len(input_dims)) / len(input_dims)
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through fusion block"""
        # Project each modality to common dimension
        projected = {}
        for modality, data in inputs.items():
            if modality in self.projectors:
                projected[modality] = self.projectors[modality](data)
        
        if not projected:
            raise ValueError("No valid modalities found in inputs")
        
        # Fuse modalities
        if self.fusion_method == 'attention':
            # Stack for attention mechanism
            modality_tensors = torch.stack(list(projected.values()), dim=1)  # [batch, modalities, hidden]
            
            # Self-attention across modalities
            fused, attention_weights = self.fusion_attention(
                modality_tensors, modality_tensors, modality_tensors
            )
            fused = self.fusion_norm(fused + modality_tensors)
            fused = fused.mean(dim=1)  # Average across modalities
            
        elif self.fusion_method == 'concatenation':
            # Simple concatenation
            fused = torch.cat(list(projected.values()), dim=-1)
            fused = self.fusion_linear(fused)
            
        elif self.fusion_method == 'weighted_sum':
            # Weighted combination
            modality_list = list(projected.values())
            weights = F.softmax(self.modality_weights, dim=0)
            fused = sum(w * tensor for w, tensor in zip(weights, modality_list))
        
        # Output projection
        output = self.output_projection(fused)
        return output


class AttentionTradingNet(nn.Module):
    """Attention-based neural network for trading decisions"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.sequence_length = config.get('sequence_length', 60)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        self.dropout = config.get('dropout', 0.1)
        
        # Input dimensions for different data types
        self.input_dims = {
            'prices': 5,        # OHLCV
            'technical': 20,    # Technical indicators
            'fundamental': 10,  # Fundamental ratios
            'sentiment': 5,     # Sentiment scores
            'alternative': 8    # Alternative data
        }
        
        # Multi-modal fusion
        self.fusion_block = MultiModalFusionBlock(
            input_dims=self.input_dims,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            fusion_method=config.get('fusion_method', 'attention')
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout)
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation='gelu'
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )
        
        # Task-specific heads
        self.price_prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)  # Next price
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 3)  # Up, Down, Sideways
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)  # Volatility prediction
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Attention visualization
        self.attention_weights = None
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the attention-based trading network"""
        batch_size, seq_len = inputs['prices'].shape[:2]
        
        # Process each timestep through fusion
        fused_sequence = []
        
        for t in range(seq_len):
            timestep_inputs = {
                modality: data[:, t, :] for modality, data in inputs.items()
                if modality in self.input_dims
            }
            fused_t = self.fusion_block(timestep_inputs)
            fused_sequence.append(fused_t)
        
        # Stack to create sequence: [seq_len, batch, hidden_dim]
        fused_sequence = torch.stack(fused_sequence, dim=0)
        
        # Add positional encoding
        fused_sequence = self.pos_encoder(fused_sequence)
        
        # Transform through encoder
        encoded_sequence = self.transformer_encoder(fused_sequence)
        
        # Store attention weights for visualization
        self.attention_weights = self._extract_attention_weights()
        
        # Use the last timestep for predictions
        final_hidden = encoded_sequence[-1]  # [batch, hidden_dim]
        
        # Multi-task predictions
        outputs = {
            'price_prediction': self.price_prediction_head(final_hidden),
            'direction_logits': self.direction_head(final_hidden),
            'volatility_prediction': self.volatility_head(final_hidden),
            'confidence': self.confidence_head(final_hidden)
        }
        
        # Add probabilities
        outputs['direction_probs'] = F.softmax(outputs['direction_logits'], dim=-1)
        
        return outputs
    
    def _extract_attention_weights(self) -> Optional[torch.Tensor]:
        """Extract attention weights for interpretation"""
        # This would extract weights from transformer layers
        # Simplified implementation
        return None
    
    def get_feature_importance(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate feature importance using gradients"""
        self.eval()
        
        # Enable gradient computation for inputs
        for modality in inputs:
            inputs[modality].requires_grad_(True)
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Calculate gradients with respect to price prediction
        loss = outputs['price_prediction'].sum()
        loss.backward()
        
        # Calculate importance scores
        importance = {}
        for modality, data in inputs.items():
            if data.grad is not None:
                importance[modality] = data.grad.abs().mean().item()
        
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance


class TransformerPredictor(nn.Module):
    """Transformer-based predictor for time series forecasting"""
    
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 1024,
                 dropout: float = 0.1, max_seq_length: int = 1000):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer layers
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through transformer predictor"""
        # x shape: [batch_size, seq_length, input_dim]
        seq_length = x.size(1)
        
        # Project to model dimension
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Transpose for transformer: [seq_len, batch, d_model]
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        if mask is not None:
            # Create attention mask
            attn_mask = self._generate_square_subsequent_mask(seq_length)
            x = self.transformer(x, mask=attn_mask, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # Get final hidden state: [batch, d_model]
        final_hidden = x[-1]
        
        # Generate predictions and uncertainty
        predictions = self.output_projection(final_hidden)
        uncertainty = self.uncertainty_head(final_hidden)
        
        return {
            'predictions': predictions,
            'uncertainty': uncertainty,
            'hidden_states': x.transpose(0, 1)  # Back to [batch, seq_len, d_model]
        }
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for transformer"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class MultiModalProcessor:
    """Main processor for multi-modal market data analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.attention_net = AttentionTradingNet(config.get('attention_config', {}))
        self.transformer_predictor = TransformerPredictor(
            input_dim=config.get('input_dim', 48),
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6)
        )
        
        # Move models to device
        self.attention_net.to(self.device)
        self.transformer_predictor.to(self.device)
        
        # Scalers for normalization
        self.scalers = {
            'prices': StandardScaler(),
            'technical': StandardScaler(),
            'fundamental': StandardScaler(),
            'sentiment': MinMaxScaler(),
            'alternative': StandardScaler()
        }
        
        # Training state
        self.training_history = []
        self.is_trained = False
        
    async def process_multimodal_data(self, data: MarketDataSequence) -> NeuralPrediction:
        """Process multi-modal market data through neural networks"""
        try:
            # Prepare inputs for attention network
            attention_inputs = {
                'prices': data.prices.to(self.device),
                'technical': data.technical.to(self.device),
                'fundamental': data.fundamental.to(self.device),
                'sentiment': data.sentiment.to(self.device),
                'alternative': data.alternative.to(self.device)
            }
            
            # Run through attention network
            with torch.no_grad():
                attention_outputs = self.attention_net(attention_inputs)
            
            # Prepare combined input for transformer
            combined_input = torch.cat([
                data.prices.flatten(start_dim=2),
                data.technical.flatten(start_dim=2),
                data.fundamental.flatten(start_dim=2),
                data.sentiment.flatten(start_dim=2),
                data.alternative.flatten(start_dim=2)
            ], dim=-1)
            
            # Run through transformer predictor
            with torch.no_grad():
                transformer_outputs = self.transformer_predictor(combined_input.to(self.device))
            
            # Combine predictions
            combined_predictions = (
                attention_outputs['price_prediction'] + transformer_outputs['predictions']
            ) / 2
            
            # Calculate confidence
            attention_confidence = attention_outputs['confidence']
            transformer_uncertainty = transformer_outputs['uncertainty']
            combined_confidence = attention_confidence * (1 / (1 + transformer_uncertainty))
            
            # Calculate prediction intervals
            std = torch.sqrt(transformer_outputs['uncertainty'])
            lower_bound = combined_predictions - 1.96 * std
            upper_bound = combined_predictions + 1.96 * std
            
            # Feature importance
            feature_importance = self.attention_net.get_feature_importance(attention_inputs)
            
            return NeuralPrediction(
                predicted_values=combined_predictions.cpu(),
                confidence_scores=combined_confidence.cpu(),
                attention_weights=self.attention_net.attention_weights,
                feature_importance=feature_importance,
                prediction_intervals=(lower_bound.cpu(), upper_bound.cpu()),
                explanation=self._generate_explanation(attention_outputs, feature_importance)
            )
            
        except Exception as e:
            logger.error(f"Error in multimodal processing: {e}")
            return NeuralPrediction(
                predicted_values=torch.zeros(1),
                confidence_scores=torch.zeros(1),
                explanation="Error in prediction"
            )
    
    async def train_models(self, train_dataset: DataLoader, 
                          val_dataset: Optional[DataLoader] = None,
                          epochs: int = 100) -> Dict[str, Any]:
        """Train the neural networks"""
        logger.info("Starting neural network training")
        
        # Optimizers
        attention_optimizer = optim.AdamW(
            self.attention_net.parameters(), 
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        transformer_optimizer = optim.AdamW(
            self.transformer_predictor.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Schedulers
        attention_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            attention_optimizer, T_max=epochs
        )
        transformer_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            transformer_optimizer, T_max=epochs
        )
        
        # Training loop
        training_history = []
        
        for epoch in range(epochs):
            # Training phase
            train_loss = await self._train_epoch(
                train_dataset, attention_optimizer, transformer_optimizer
            )
            
            # Validation phase
            val_loss = 0.0
            if val_dataset:
                val_loss = await self._validate_epoch(val_dataset)
            
            # Update learning rates
            attention_scheduler.step()
            transformer_scheduler.step()
            
            # Record history
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'attention_lr': attention_scheduler.get_last_lr()[0],
                'transformer_lr': transformer_scheduler.get_last_lr()[0]
            }
            training_history.append(epoch_metrics)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}")
        
        self.training_history = training_history
        self.is_trained = True
        
        logger.info("Neural network training completed")
        return {
            'training_history': training_history,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss
        }
    
    async def _train_epoch(self, dataloader: DataLoader, 
                          attention_opt: optim.Optimizer,
                          transformer_opt: optim.Optimizer) -> float:
        """Train for one epoch"""
        self.attention_net.train()
        self.transformer_predictor.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            attention_opt.zero_grad()
            transformer_opt.zero_grad()
            
            # Unpack batch
            inputs, targets = batch
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = targets.to(self.device)
            
            # Forward passes
            attention_outputs = self.attention_net(inputs)
            
            # Prepare combined input for transformer
            combined_input = self._prepare_transformer_input(inputs)
            transformer_outputs = self.transformer_predictor(combined_input)
            
            # Calculate losses
            attention_loss = self._calculate_attention_loss(attention_outputs, targets)
            transformer_loss = F.mse_loss(transformer_outputs['predictions'], targets[:, -1:])
            
            # Combined loss
            total_batch_loss = attention_loss + transformer_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.attention_net.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.transformer_predictor.parameters(), 1.0)
            
            # Optimizer steps
            attention_opt.step()
            transformer_opt.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    async def _validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch"""
        self.attention_net.eval()
        self.transformer_predictor.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = targets.to(self.device)
                
                # Forward passes
                attention_outputs = self.attention_net(inputs)
                combined_input = self._prepare_transformer_input(inputs)
                transformer_outputs = self.transformer_predictor(combined_input)
                
                # Calculate losses
                attention_loss = self._calculate_attention_loss(attention_outputs, targets)
                transformer_loss = F.mse_loss(transformer_outputs['predictions'], targets[:, -1:])
                
                total_batch_loss = attention_loss + transformer_loss
                total_loss += total_batch_loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _calculate_attention_loss(self, outputs: Dict[str, torch.Tensor], 
                                targets: torch.Tensor) -> torch.Tensor:
        """Calculate multi-task loss for attention network"""
        # Price prediction loss
        price_loss = F.mse_loss(outputs['price_prediction'], targets[:, -1:])
        
        # Direction classification loss (if available)
        direction_loss = torch.tensor(0.0, device=self.device)
        if 'direction_logits' in outputs and targets.size(1) > 1:
            direction_targets = (targets[:, -1] > targets[:, -2]).long()
            direction_loss = F.cross_entropy(outputs['direction_logits'], direction_targets)
        
        # Volatility loss
        volatility_loss = torch.tensor(0.0, device=self.device)
        if 'volatility_prediction' in outputs:
            # Calculate target volatility from price changes
            if targets.size(1) > 5:
                returns = targets.diff(dim=1)
                target_vol = returns[:, -5:].std(dim=1, keepdim=True)
                volatility_loss = F.mse_loss(outputs['volatility_prediction'], target_vol)
        
        # Combined loss
        total_loss = price_loss + 0.1 * direction_loss + 0.1 * volatility_loss
        return total_loss
    
    def _prepare_transformer_input(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare combined input for transformer"""
        return torch.cat([
            inputs['prices'].flatten(start_dim=2),
            inputs['technical'].flatten(start_dim=2),
            inputs['fundamental'].flatten(start_dim=2),
            inputs['sentiment'].flatten(start_dim=2),
            inputs['alternative'].flatten(start_dim=2)
        ], dim=-1)
    
    def _generate_explanation(self, attention_outputs: Dict[str, torch.Tensor],
                            feature_importance: Dict[str, float]) -> str:
        """Generate human-readable explanation of the prediction"""
        direction_probs = attention_outputs.get('direction_probs')
        confidence = attention_outputs.get('confidence', torch.tensor([0.5]))
        
        # Direction prediction
        if direction_probs is not None:
            direction_idx = torch.argmax(direction_probs, dim=-1).item()
            direction_names = ['Down', 'Sideways', 'Up']
            predicted_direction = direction_names[direction_idx]
            direction_confidence = direction_probs[0, direction_idx].item()
        else:
            predicted_direction = "Unknown"
            direction_confidence = 0.5
        
        # Top features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        feature_text = ", ".join([f"{name} ({importance:.1%})" for name, importance in top_features])
        
        explanation = f"""
        Prediction: {predicted_direction} direction with {direction_confidence:.1%} confidence.
        Overall model confidence: {confidence.item():.1%}
        Key contributing factors: {feature_text}
        """
        
        return explanation.strip()
    
    def save_models(self, path_prefix: str) -> None:
        """Save trained models"""
        torch.save(self.attention_net.state_dict(), f"{path_prefix}_attention_net.pth")
        torch.save(self.transformer_predictor.state_dict(), f"{path_prefix}_transformer.pth")
        logger.info(f"Models saved with prefix: {path_prefix}")
    
    def load_models(self, path_prefix: str) -> None:
        """Load trained models"""
        self.attention_net.load_state_dict(torch.load(f"{path_prefix}_attention_net.pth", map_location=self.device))
        self.transformer_predictor.load_state_dict(torch.load(f"{path_prefix}_transformer.pth", map_location=self.device))
        self.is_trained = True
        logger.info(f"Models loaded from prefix: {path_prefix}")


class TradingDataset(Dataset):
    """Dataset for training neural networks on trading data"""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 60, 
                 prediction_horizon: int = 1):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Prepare features and targets
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare input features and target values"""
        self.features = {}
        
        # Price features (OHLCV)
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        self.features['prices'] = self.data[price_columns].values
        
        # Technical indicators
        tech_columns = [col for col in self.data.columns if any(indicator in col.lower() 
                       for indicator in ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr'])]
        if tech_columns:
            self.features['technical'] = self.data[tech_columns].values
        else:
            self.features['technical'] = np.zeros((len(self.data), 20))
        
        # Fundamental features (placeholder)
        self.features['fundamental'] = np.random.randn(len(self.data), 10) * 0.1
        
        # Sentiment features (placeholder)
        self.features['sentiment'] = np.random.randn(len(self.data), 5) * 0.1
        
        # Alternative data features (placeholder)
        self.features['alternative'] = np.random.randn(len(self.data), 8) * 0.1
        
        # Target values (future prices)
        self.targets = self.data['close'].values
        
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        # Input sequences
        inputs = {}
        for modality, data in self.features.items():
            inputs[modality] = torch.FloatTensor(
                data[idx:idx + self.sequence_length]
            ).unsqueeze(0)  # Add batch dimension for each modality
        
        # Target values
        target_idx = idx + self.sequence_length + self.prediction_horizon - 1
        targets = torch.FloatTensor(
            self.targets[idx + self.sequence_length - 1:target_idx + 1]
        )
        
        return inputs, targets


# Example usage and testing
async def demo_neural_networks():
    """Demonstrate neural network capabilities"""
    logger.info("Starting neural networks demo")
    
    # Configuration
    config = {
        'attention_config': {
            'sequence_length': 60,
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'fusion_method': 'attention'
        },
        'input_dim': 48,  # Combined input dimension
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'learning_rate': 0.001,
        'weight_decay': 0.01
    }
    
    # Initialize processor
    processor = MultiModalProcessor(config)
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': np.random.randn(len(dates)) * 2 + 100,
        'high': np.random.randn(len(dates)) * 2 + 102,
        'low': np.random.randn(len(dates)) * 2 + 98,
        'close': np.random.randn(len(dates)) * 2 + 100,
        'volume': np.random.lognormal(15, 0.5, len(dates)),
        'sma_20': np.random.randn(len(dates)) * 1 + 100,
        'rsi': np.random.randn(len(dates)) * 10 + 50,
        'macd': np.random.randn(len(dates)) * 0.5,
    }, index=dates)
    
    # Create dataset
    dataset = TradingDataset(sample_data, sequence_length=60)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test forward pass with sample data
    sample_batch = next(iter(dataloader))
    inputs, targets = sample_batch
    
    # Create MarketDataSequence for processing
    sequence = MarketDataSequence(
        prices=inputs['prices'],
        technical=inputs['technical'],
        fundamental=inputs['fundamental'],
        sentiment=inputs['sentiment'],
        alternative=inputs['alternative'],
        timestamps=[datetime.now()],
        symbols=['DEMO'],
        metadata={}
    )
    
    # Process through neural networks
    prediction = await processor.process_multimodal_data(sequence)
    
    logger.info(f"Neural Network Results:")
    logger.info(f"Predicted Value: {prediction.predicted_values.item():.4f}")
    logger.info(f"Confidence: {prediction.confidence_scores.item():.4f}")
    logger.info(f"Feature Importance: {prediction.feature_importance}")
    logger.info(f"Explanation: {prediction.explanation}")
    
    # Quick training demo (reduced epochs for demo)
    logger.info("Starting quick training demo...")
    training_results = await processor.train_models(dataloader, epochs=10)
    logger.info(f"Training completed. Final loss: {training_results['final_train_loss']:.6f}")
    
    logger.info("Neural networks demo completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_neural_networks())