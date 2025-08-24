"""
Phase 6: Advanced AI/ML Trading System
Breakthrough AI capabilities for next-generation trading

This module implements state-of-the-art AI/ML systems including:
- Reinforcement Learning Agents (DQN, PPO, A3C)
- Large Language Models for Market Analysis
- Computer Vision for Alternative Data
- Advanced Neural Architectures
- AI-Driven Portfolio Optimization
"""

from .rl_agents import TradingRLAgent, DQNAgent, PPOAgent, A3CAgent
from .llm_integration import LLMMarketAnalyzer, SentimentAnalyzer, NewsProcessor
from .computer_vision import ChartPatternDetector, SatelliteImageryAnalyzer, CVDataProcessor
from .neural_networks import MultiModalProcessor, AttentionTradingNet, TransformerPredictor
from .portfolio_optimizer import DeepRLPortfolioOptimizer, QuantumInspiredOptimizer
from .model_lifecycle import AIModelManager, ContinuousLearningPipeline

__version__ = "1.0.0"
__author__ = "Advanced AI/ML Research Engineer"
__status__ = "Production"

__all__ = [
    # Reinforcement Learning
    "TradingRLAgent", "DQNAgent", "PPOAgent", "A3CAgent",
    
    # LLM Integration
    "LLMMarketAnalyzer", "SentimentAnalyzer", "NewsProcessor",
    
    # Computer Vision
    "ChartPatternDetector", "SatelliteImageryAnalyzer", "CVDataProcessor",
    
    # Neural Networks
    "MultiModalProcessor", "AttentionTradingNet", "TransformerPredictor",
    
    # Portfolio Optimization
    "DeepRLPortfolioOptimizer", "QuantumInspiredOptimizer",
    
    # Model Lifecycle
    "AIModelManager", "ContinuousLearningPipeline"
]